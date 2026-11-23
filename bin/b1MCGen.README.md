# b1MCGen

`b1MCGen` 是 [b1MCGen.py](b1MCGen.py) 的命令行入口，管理 Belle MC 的生成和 GSIM 重建任务。它本身不定义物理过程：生成器、decay card、束流条件和输出格式由 `--model_file` 指定的 basf2 steering file 决定。使用 BelleMCOutput 输出生成级 Belle 文件，再使用 basf/Geant3 模拟，才能得到 Belle 探测器模拟的 MDST；B2BII 转换本身不是探测器模拟。参见 [Belle II 官方文档](https://software.belle2.org/development/sphinx/b2bii/doc/BelleDataMC.html#generating-mc-sample)。

**当前版本不建议直接用于生产。** GSIM 任务间随机数独立性尚未验证；`--dry_run` 仍会创建文件。

## A. 依赖和输入

- Python 3、LSF `bsub`、可运行的 basf2 及 Belle basf/GSIM 环境。早期 experiment 的 `--sim` 还需要站点的 `/sw/belle/local/bin/centos7-exec` 容器及工作节点上的 `/tmp` 空间。
- `--sim` 依赖与 `b1MCGen.py` 同目录的 [b1MCGen_gsim_runner.py](b1MCGen_gsim_runner.py)：它由提交的模拟作业自动调用，负责运行与读回验证。通过符号链接启动 `b1MCGen` 时，程序从链接指向的实际脚本目录查找 runner；该文件必须在计算节点可读，且在模拟作业完成前不要移动或删除。
- `--base_dir` 默认是 `/gpfs/group/belle2/users2022/wangz/SignalMC_belle1/database`，其中需要：
  - `background_files_merged.dat`：每行 `exp run_min run_max background_path`。
  - `run_database_energy.dat`：每行 `exp run dataset luminosity ecms`。
- 背景文件必须在任务运行节点可读。当前映射可能是一个背景文件对应多个 run，并非严格逐 run 背景。
- 已配置好的 basf2 生成 steering file 须接受依次为 `exp run event_count background_file output_file` 的参数，并输出可供 Belle basf 读取的生成级文件。
- GSIM 配置针对使用 `b2bii_beamParameters_with_smearing` 的 BelleMCOutput 输入，已移除 `bpsmear`，以免重复展宽。其他生成格式或未展宽的输入需另行配置和验证。

程序在当前目录写 `belle1_mc_gen.log`，并在数据库目录 `.cache/` 创建或读取 pickle 缓存。缓存不会自动检测源文件更新；更换数据库内容后须自行处理旧缓存。不要读取不可信的 pickle 缓存。

### A-1. 将 Belle II steering file 改成 Belle I 生成脚本

以 [exp 55/run 3 的 `generate.py`](/gpfs/group/belle2/users2022/wangz/SignalMC_belle1_new/pilot_jpsi_phiKK_e55_r3_1000/generate.py) 为例：**保留生成过程，替换束流条件和输出，把探测器模拟移到下一阶段。**

| 原 Belle II 脚本 | Belle I 脚本怎么改 |
| --- | --- |
| `add_beamparameters(main, "Y4S")` | 不再手动指定 Belle II 束流。用 `EventInfoSetter` 指定 Belle 的 exp/run，并从 `b2bii_beamParameters_with_smearing` 读取相应条件。后续 GSIM 不再加 `bpsmear`。 |
| 生成器和 decay card | 此例保留 `EvtGenInput`、`ParentParticle="vpho"` 及原始 VECTORISR `decay.dec`，不额外启用 PHOTOS。若原样本使用 PHOKHARA，保留其生成器与过程设置，不要换成下面的 EvtGen 示例。ISR 可探测样本还需核对角度所在参考系。 |
| `add_simulation`、Belle II 背景叠加、`add_reconstruction`、`add_mdst_output` | 从 basf2 路径中移除。生成器后接 `BelleMCOutput` 写 **生成级 `.dat`**；待生成完成，再用 `b1MCGen --sim` 运行 Belle basf/Geant3、叠加 Belle 背景并重建 **`.mdst`**。 |

**用作 `--model_file` 时不要直接使用 pilot 脚本**：它写死了 exp 55、run 3、1000 个事件和输出路径。下面是对应的参数化版本；将原始 `decay.dec` 放在 steering file 同一目录。`b1MCGen` 传入的第 4 个参数是背景文件，**不是 decay card**；这里命名为 `_background_file` 仅为保持调用接口，生成阶段不读取或叠加背景，输出的 `.dat` 是信号生成级文件。

背景在后续 `--sim` 阶段处理：`b1MCGen` 从生成文件名识别 exp/run，重新通过数据库查找对应背景文件，将其设置为 `ADDBG_DAT`，由 Belle BASF 的 `addbg` 模块在模拟时叠加。模拟时应使用与生成时一致的数据库和筛选条件；背景映射可能覆盖多个 run，并非严格逐 run。

```python
import sys
from pathlib import Path
import basf2 as b2

exp, run, n, _background_file, output_dat = sys.argv[1:]
card = Path(__file__).resolve().with_name("decay.dec")

b2.conditions.metadata_providers = ["/sw/belle/b2bii/database/conditions/b2bii.sqlite"]
b2.conditions.payload_locations = ["/sw/belle/b2bii/database/conditions/"]
b2.conditions.override_globaltags()
b2.conditions.prepend_globaltag("b2bii_beamParameters_with_smearing")

path = b2.create_path()
path.add_module("EventInfoSetter", expList=int(exp), runList=int(run), evtNumList=int(n))
path.add_module("EvtGenInput", ParentParticle="vpho", userDECFile=str(card))
path.add_module("BelleMCOutput", outputFileName=output_dat)
b2.process(path, calculateStatistics=True)
print(b2.statistics)
```

## B. 命令和输出

运行 `b1MCGen --help` 查看完整参数。默认不限制 experiment 或 dataset，因此包括数据库中的所有能区；使用 `--exps` 和 `--dts` 显式选择。`--n` 是按所选 run 的 luminosity 分配的总事件数，默认为 10000。整数事件数按最大余数法分配，保持总数准确且不会产生负数；这可能与旧版逐 run 四舍五入的结果稍有不同。

| 选项 | 行为 |
| --- | --- |
| `--ava_exps`、`--ava_dts`、`--ava_run` | 显示所选 experiment、dataset 或 run。 |
| `--exp_events` | 显示按 luminosity 分配的每个 experiment 事件数。 |
| `--gen` | 按 experiment 和背景文件合并 run，提交 basf2 生成任务。每个合并任务使用该任务最小 run 的条件；单个 run 超过 9999 时会拆分。 |
| `--gen_by_run` | 每个有事件的 run 提交一个生成任务。 |
| `--skip_exp19` | 在分配 `--n` 前排除 exp 19；生成和模拟均跳过它。默认不排除，不会修改已有样本。其他早期 experiment 仍可能需要同样的 32 位依赖。 |
| `--gen_exp_run --exp E --run R --n_run N` | 为指定 experiment/run 提交一个生成任务，单任务上限 9999。 |
| `--sim` | 扫描 `top_dir/gen` 下匹配 `e*_r*.dat` 的文件，创建脚本并提交 basf/GSIM 任务。 |
| `--queue Q` | LSF 队列，默认为 `s`。 |
| `--dry_run` | 不调用 `bsub`；仍会创建目录、日志、缓存以及 GSIM 脚本。 |

生成级文件及日志位于 `top_dir/gen/`（逐 run 模式在 `gen/e<exp>/`）；模拟输出和日志位于 `top_dir/sim/`，脚本位于 `top_dir/sim_script/`。`--sim` 保留生成文件相对于 `gen/` 的路径和完整文件名：例如 `gen/a/e55_r3_j0.dat` 对应 `sim/a/e55_r3_j0.mdst`、`sim/a/e55_r3_j0_gsim.log` 及 `sim_script/a/e55_r3_j0_gsim.*`。不同输入不会共享同一个 MDST 或脚本。生成和模拟是分开的阶段，必须等生成作业完成后才运行模拟。

### B-1. 具体步骤：单个 run 的 1000 事件测试

以下示例依次提交生成和模拟作业，使用 exp 55/run 3、`s` 队列和 1000 个事件。**不要直接用于已有样本目录**；将 `MODEL` 改为 A-1 中保存的参数化脚本的绝对路径，并确保它旁边有该样本的 `decay.dec`。pilot 的 `generate.py` 参数是写死的，不能直接用作 `MODEL`。不同物理样本须各用独立目录、对应的生成器配置和 decay card。

1. 在提交节点设置环境并创建一个**不存在的**工作目录：

   ```bash
   source /cvmfs/belle.cern.ch/tools/b2setup release-11-00-00

   MODEL=/absolute/path/to/your/generate.py
   WORK=/gpfs/group/belle2/users2022/wangz/SignalMC_belle1_new/test_e55_r3_1000
   test -f "$MODEL" && test -f "$(dirname "$MODEL")/decay.dec"
   mkdir "$WORK"  # 如果目录已存在会报错；换新目录，不覆盖已有文件
   ```

2. 提交生成（可通过`--dry-run`）预览。 `--n_run` 是这个指定 run 的事件数，和按 luminosity 分配的 `--n` 不同：

   ```bash
   b1MCGen --gen_exp_run --exp 55 --run 3 --n_run 1000 \
     --exps 55 --skip_exp19 --queue s --model_file "$MODEL" --top_dir "$WORK"
   ```

   从提交信息记录 LSF job ID。等待作业 `DONE` 后核对生成日志和输出；如果是 `EXIT`、缺文件或事件数不对，先排查，不要继续模拟：

   ```bash
   grep 'BelleMCOutput' "$WORK/gen/e55_r3_manual.log"
   ```

   确认日志中的 `BelleMCOutput` 处理了 1000 个事件。`gen/*.dat` 是生成级文件，不是最终 MDST。

3. 提交 GSIM：**必须等生成作业完成并检查通过**。本次不要先在同一 `WORK` 上运行 `--sim --dry_run`，因为它会创建脚本，之后真实提交会因防覆盖检查而拒绝。

   ```bash
   b1MCGen --sim --exps 55 --skip_exp19 --queue s --top_dir "$WORK"
   ```

   记录新的模拟作业 ID；等待其完成后检查：

   ```bash
   grep -aE 'ERROR:|Verified exp|event_process terminated' \
     "$WORK/sim/e55_r3_manual_gsim.log"
   tail -20 "$WORK/sim_script/e55_r3_manual_input_check.log"
   tail -20 "$WORK/sim_script/e55_r3_manual_output_check.log"
   ```

   成功日志应含 `Verified exp 55, run 3: 1000 events`，输入和输出读回均应显示 exp 55/run 3、1000 个事件。若作业失败或只有 `.partial`，不要把它当作成功产物。生成作业 ID 和模拟作业 ID 不同，均须分别检查。

### B-2. 按 luminosity 分配到多个 run

为每个样本准备自己的 `MODEL` 和**新** `WORK`。下面以一个样本总共 100000 个事件为例，排除 exp 19；不写 `--exps` 和 `--dts` 时包含数据库中其余所有 experiment 和能区：

```bash
b1MCGen --gen --skip_exp19 --n 100000 --queue s \
  --model_file "$MODEL" --top_dir "$WORK"
```

如需限制范围，可在两条命令中同样加入 `--exps 55 --dts 4S`。等待**所有**生成作业完成，检查各生成日志和事件数之和；然后提交 `b1MCGen --sim --skip_exp19 --queue s --top_dir "$WORK"`，如果生成时使用了 `--exps` 或 `--dts`，这里也使用相同筛选。模拟提交后逐项检查 LSF 状态、`sim/*_gsim.log` 和读回日志。不同样本分别执行，不能共用一个 `WORK`。

`--sim` 在提交前检查所有目标；已存在输出、索引、锁、日志或脚本时会报错退出。脚本以独占方式创建，任务运行时还会创建 `*.mdst.lock` 目录，防止同一任务重复执行。锁在失败或成功后保留。`--sim --dry_run` 也会创建脚本，因此不能在同一目录直接再次执行 `--sim`：请使用独立测试目录，或在确认没有任务运行且不再需要相关文件后，手动清理对应脚本、锁和输出。不要删除其他用户数据。

每个 GSIM 任务先用 64 位 Belle BASF 读回生成文件，核对 exp、run 和事件数，再执行模拟，并读回 MDST 核对同样的信息。BASF 主进程成功且输出比输入少 1–3 个事件时，即使检测到工作进程失败，也会发布 `.mdst` 并输出 `WARNING`，记录 exp、run、expected、observed、missing 和是否检测到工作进程失败；事件数一致且无工作进程失败时正常发布。输出多于输入、少于输入超过 3 个、读回失败、主进程失败或工作进程失败但事件数一致时作业失败。失败时后期实验的 `.partial` 文件和日志保留用于排查。读回日志位于 `sim_script/` 中的 `*_input_check.log` 和 `*_output_check.log`；GSIM 日志位于 `sim/`。此检查只能比较实际输入和输出，**不能证明生成文件包含原计划的事件数**，仍需检查生成日志；也不能证明事件一一对应或替代物理分布及随机数验证。已有失败的 `.partial` 不会自动重新验证或发布；提交的作业运行时读取共享 runner，修改会影响尚未启动的作业。

BASF 输入、输出和脚本路径不能包含空白或反斜线；GSIM 参数文件的文件名最长为 132 字节。提交失败会报错退出，但之前已提交的任务不会自动取消；检查日志和 LSF 状态后再决定如何恢复。

## C. 使用限制

- GSIM 输入中的 `RNDM` 卡仍固定；GSIM 每个事件会从 BASF RANECU 获取种子，不能仅通过改变此卡保证任务随机数独立。不同任务的随机数独立性尚未验证，不能用于正式生产。
- 合并 run 模式把每个任务的事件按其最小 run 生成，而不是逐 run 使用束流条件。`--gen_by_run` 仍按 run 提交，不会自动拆分超过 9999 事件的 run。
- 生成阶段提交失败、缺失输入和缓存解析错误可能只写日志，未必使整个程序以非零状态退出。检查各任务日志及实际事件数，不要仅凭提交成功判断产出成功。
- 在 `cw04` 和 `cb054` 的宿主环境，早期 32 位 basf 缺少 `libncurses.so.5` 和 `libg2c.so.0`；部分 GPFS 文件的 inode 超出 32 位 `stat` 范围。`--sim` 对 exp ≤27 自动将输入暂存到节点 `/tmp` 并用站点 CentOS 7 容器执行 GSIM，成功校验后复制并发布 MDST；需要足够的本地临时空间。不要把 EL9 的 basf2 生成阶段一起放入容器。
- 此脚本不会自动选择对应 PHOKHARA/VECTORISR 参数、做 Belle 实验室系角度变换，也不会保证 PHOTOS 设置与原样本一致；这些由生成 steering file 和 decay card 决定。

已用本程序生成的提交脚本在 LSF 验证 exp 7/run 6 的 68 个事件和 exp 45/run 32 的 48 个事件，输入和 MDST 均成功读回并一致。其他 experiment、能区及生成器尚未因此得到验证。
