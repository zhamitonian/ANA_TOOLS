# Fit Framework 重构设计

## 核心理念

**关键洞察**: 不同的拟合类型（1D resonance fit, 2D fit, chi-square fit等）之间的主要区别仅在于：
1. **变量维度**（1D, 2D, 多维）
2. **PDF配置**（具体使用哪些PDF）
3. **Model结构**（如何组合这些PDF：乘积、卷积、求和等）
4. **绘图方式**（1D投影还是多维投影）

其它部分（数据集创建、拟合流程、sPlot、结果保存等）**完全相同**！

### 设计演进

**v1**: ~~区分Signal和Background PDF~~ → **已废弃**  
- 最初设计区分signal_pdfs和background_pdfs
- 发现这是人为的分类，本质上都是PDF

**v2**: **统一PDF + 灵活Model结构** ← **当前设计**
- 所有PDF一视同仁，不区分signal/background
- 在model字符串中直接表达所有操作（PROD, FCONV, 括号等）
- 无需单独的operations参数

**核心策略**：
- **统一使用 `GenericFit` 类处理所有拟合**
- **专注于扩展 PDF Builders**
- **通过model字符串表达复杂结构**

## 架构设计

```
拟合框架
├── pdf_builders.py          # PDF构建器集合（核心扩展点）
│   ├── PDFBuilder           # 抽象基类
│   ├── PDF Builders         # 具体实现（无signal/background之分）
│   │   ├── BreitWignerGaussBuilder
│   │   ├── CrystalBallBuilder
│   │   ├── VoigtianBuilder
│   │   ├── DoubleGaussianBreitWignerBuilder
│   │   ├── GaussianBuilder
│   │   ├── PolynomialBuilder
│   │   ├── ChebychevBuilder
│   │   ├── ArgusBGBuilder
│   │   ├── ExponentialBuilder
│   │   ├── FlatBuilder
│   │   └── ...（易于添加新的）
│   └── PDFBuilderRegistry   # 统一管理所有builders
│
├── model_parser.py          # Model结构解析器（新增）
│   └── ModelParser          # 解析复杂model字符串
│
├── generic_fit.py           # 通用拟合类（适用所有场景）
│   └── GenericFit           # 单一类处理所有拟合
│
└── fit_tools.py            # 工具函数
```

## 使用示例

### Example 1: 简单的1D共振峰拟合

```python
from FIT.generic_fit import GenericFit

fit = GenericFit(
    tree=tree,
    output_dir="output/",
    
    # 定义变量：(name, min, max)
    variables=[("phi_M", 1.0, 1.06)],
    
    # PDF配置（无signal/background之分）
    pdfs=[
        {
            "name": "sig",                    # 自定义PDF名称
            "var": "phi_M",
            "type": "bw_gauss",               # Breit-Wigner ⊗ Gaussian
            "config": {
                "mass": 1.0195, 
                "width": 0.004249
            }
        },
        {
            "name": "bkg",                    # 自定义PDF名称
            "var": "phi_M",
            "type": "polynomial",             # 多项式背景
            "config": {"order": 1}
        }
    ],
    
    # Model结构：直接用PDF名称
    model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg"
)

result, fit_results = fit.run()
print(f"nsig = {fit_results['nsig']:.2f} ± {fit_results['nsig_err']:.2f}")
```

**关键点**：
- 每个PDF必须指定唯一的`name`
- `model`字符串中直接使用这些名称
- 不再区分signal和background

### Example 2: 2D拟合（直接使用PROD操作）

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("phi1_M", 1.0, 1.06), ("phi2_M", 1.0, 1.06)],
    
    # 为每个维度定义PDF
    pdfs=[
        {"name": "sig1", "var": "phi1_M", "type": "bw_gauss", 
         "config": {"mass": 1.0195, "width": 0.004249}},
        {"name": "sig2", "var": "phi2_M", "type": "bw_gauss", 
         "config": {"mass": 1.0195, "width": 0.004249}},
        {"name": "bkg1", "var": "phi1_M", "type": "polynomial", 
         "config": {"order": 1}},
        {"name": "bkg2", "var": "phi2_M", "type": "polynomial", 
         "config": {"order": 1}}
    ],
    
    # Model结构：直接在字符串中使用PROD操作！
    model=(
        "nsig[100,0,5000]*PROD(sig1, sig2) + "           # 双信号
        "nbkg1[50,0,2000]*PROD(bkg1, bkg2) + "           # 双背景
        "nbkg2[50,0,2000]*PROD(sig1, bkg2) + "           # 混合1
        "nbkg3[50,0,2000]*PROD(bkg1, sig2)"              # 混合2
    ),
    
    plot_vars=["phi1_M", "phi2_M"]
)

result, fit_results = fit.run()
```

**关键点**：
- 直接在model字符串中使用`PROD(pdf1, pdf2)`表达乘积
- 无需单独定义operations参数
- 解析器自动创建中间PDF

### Example 3: 卷积拟合（FCONV）

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("mass", 1.0, 1.06)],
    
    pdfs=[
        {"name": "bw", "var": "mass", "type": "voigtian",
         "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.0}},
        {"name": "resolution", "var": "mass", "type": "gaussian",
         "config": {"mean": 0.0, "sigma": 0.001}},
        {"name": "bkg", "var": "mass", "type": "exponential",
         "config": {"tau": -1.0}}
    ],
    
    # FCONV直接在model字符串中！
    model="nsig[100,0,10000]*FCONV(mass, bw, resolution) + nbkg[50,0,10000]*bkg"
)

result, fit_results = fit.run()
```

**卷积语法**：`FCONV(variable, pdf1, pdf2)`

### Example 4: 复杂表达式（括号分组）

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("mass", 1.0, 1.06)],
    
    pdfs=[
        {"name": "sig1", "var": "mass", "type": "gaussian",
         "config": {"mean": 1.0195, "sigma": 0.001}},
        {"name": "sig2", "var": "mass", "type": "gaussian",
         "config": {"mean": 1.0200, "sigma": 0.002}},
        {"name": "bkg", "var": "mass", "type": "polynomial",
         "config": {"order": 2}}
    ],
    
    # 使用括号分组：两个信号的和
    model="nsig[100,0,10000]*(sig1 + sig2) + nbkg[50,0,10000]*bkg"
)
```

**括号规则**：
- 用括号组合多个PDF：`(pdf1 + pdf2)`
- 解析器自动创建SUM PDF

### Example 5: 3D拟合

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[
        ("var1", 1.0, 1.06),
        ("var2", 1.0, 1.06),
        ("var3", 0, 50)
    ],
    
    pdfs=[
        {"name": "s1", "var": "var1", "type": "bw_gauss", "config": {...}},
        {"name": "s2", "var": "var2", "type": "bw_gauss", "config": {...}},
        {"name": "s3", "var": "var3", "type": "gaussian", "config": {...}},
        {"name": "b1", "var": "var1", "type": "polynomial", "config": {"order": 1}},
        {"name": "b2", "var": "var2", "type": "polynomial", "config": {"order": 1}},
        {"name": "b3", "var": "var3", "type": "exponential", "config": {}}
    ],
    
    # 3D乘积
    model=(
        "nsig[100,0,5000]*PROD(s1, s2, s3) + "
        "nbkg[50,0,2000]*PROD(b1, b2, b3)"
    ),
    
    plot_vars=["var1", "var2", "var3"]
)
```

### Example 6: 超级复杂的混合操作

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("mass", 1.0, 1.06)],
    
    pdfs=[
        {"name": "bw", "var": "mass", "type": "voigtian", "config": {...}},
        {"name": "gauss", "var": "mass", "type": "gaussian", "config": {...}},
        {"name": "cb", "var": "mass", "type": "crystal_ball", "config": {...}},
        {"name": "bkg", "var": "mass", "type": "chebychev", "config": {"order": 2}}
    ],
    
    # 卷积 + Crystal Ball（括号内）+ 背景
    model="nsig[100,0,10000]*(FCONV(mass, bw, gauss) + cb) + nbkg[50,0,10000]*bkg"
)
```

**支持的操作**：
- `PROD(pdf1, pdf2, ...)` - 乘积
- `FCONV(var, pdf1, pdf2)` - 卷积
- `(pdf1 + pdf2)` - 求和（自动创建SUM）
- 任意嵌套组合

## Model字符串语法

### 基本语法

```
model = "yield_var[initial,min,max] * pdf_expression + ..."
```

### 支持的操作

| 操作 | 语法 | 说明 | 示例 |
|------|------|------|------|
| 简单引用 | `pdf_name` | 直接使用PDF名称 | `nsig[100,0,1000]*sig` |
| 乘积 | `PROD(pdf1, pdf2, ...)` | 多个PDF的乘积 | `PROD(sig1, sig2)` |
| 卷积 | `FCONV(var, pdf1, pdf2)` | PDF1与PDF2的卷积 | `FCONV(mass, bw, gauss)` |
| 求和 | `(pdf1 + pdf2)` | 括号内的求和 | `(gauss1 + gauss2)` |
| 嵌套 | 任意组合 | 支持无限嵌套 | `PROD(FCONV(...), pdf3)` |

### Model字符串示例

```python
# 1. 简单模型
"nsig[100,0,1000]*sig + nbkg[50,0,500]*bkg"

# 2. 2D模型（乘积）
"nsig[100,0,5000]*PROD(sig1, sig2) + nbkg[50,0,2000]*PROD(bkg1, bkg2)"

# 3. 卷积模型
"nsig[100,0,1000]*FCONV(mass, bw, gauss) + nbkg[50,0,500]*exponential"

# 4. 复杂组合
"nsig[100,0,1000]*(FCONV(mass, bw, gauss) + crystal_ball) + nbkg[50,0,500]*poly"

# 5. 3D模型
"nsig[100,0,5000]*PROD(pdf1, pdf2, pdf3) + nbkg[50,0,2000]*PROD(bkg1, bkg2, bkg3)"

# 6. 多背景成分
"nsig[100,0,1000]*sig + nbkg1[50,0,500]*bkg1 + nbkg2[30,0,300]*bkg2"
```

### 解析器工作原理

`ModelParser`会自动：
1. **提取yield变量**：识别`nsig[...]`, `nbkg[...]`等
2. **识别操作**：PROD, FCONV, 括号等
3. **创建中间PDF**：为复杂操作自动生成临时PDF
4. **生成RooFit代码**：转换为workspace.factory()调用

示例：
```python
# 输入
model = "nsig[100,0,1000]*PROD(sig1, sig2) + nbkg[50,0,500]*bkg"

# 解析器内部会：
# 1. workspace.factory("nsig[100,0,1000]")
# 2. workspace.factory("PROD::tmp_prod_1(sig1, sig2)")
# 3. workspace.factory("SUM::model(nsig*tmp_prod_1, nbkg*bkg)")
```

## 轻松切换不同的PDF组合

```python
# 系统误差研究：对比不同的Signal PDF
for sig_type in ["bw_gauss", "voigtian", "crystal_ball", "double_gauss_bw"]:
    fit = GenericFit(
        tree=tree,
        output_dir=f"output/sys_{sig_type}",
        variables=[("phi_M", 1.0, 1.06)],
        
        # 只需改变type参数！
        pdfs=[
            {"name": "sig", "var": "phi_M", "type": sig_type,
             "config": {"mass": 1.0195, "width": 0.004249}},
            {"name": "bkg", "var": "phi_M", "type": "polynomial",
             "config": {"order": 1}}
        ],
        
        model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg"
    )
    
    result, fit_results = fit.run()
```

## 绘图配置

### 基本配置

所有变量默认都会绘图。通过`plot_config`字典配置绘图细节：

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("phi_M", 1.0, 1.06)],
    pdfs=[...],
    model="...",
    
    # 绘图配置
    plot_config={
        # 基本设置
        "nbin": 60,  # 直方图bin数
        
        # 坐标轴标签（每个变量可以有不同的标签）
        "xlabel": {
            "phi_M": "M_{K^{+}K^{-}} (GeV/c^{2})",
            "phi2_M": "M_{K^{+}K^{-}}^{(2)} (GeV/c^{2})"
        },
        "ylabel": {
            "phi_M": "Candidates / (1 MeV/c^{2})",
            # 如果不指定，会自动计算
        },
        
        # 要绘制的PDF组件及其样式
        "components": {
            "sig": {  # PDF名称（在workspace中的名称）
                "label": "Signal",  # 图例中的标签
                "color": ROOT.kRed,  # 线条颜色
                "style": 4,  # 线型（1=实线, 2=虚线, 4=点划线等）
                "width": 3   # 线宽
            },
            "bkg": {
                "label": "Background",
                "color": ROOT.kGreen+2,
                "style": 7,
                "width": 3
            }
        },
        
        # 图例配置
        "legend": {
            "x1": 0.65, "x2": 0.95,  # 水平位置
            "y2": 0.9,                # 顶部位置（底部自动计算）
            "show_chi2": True,        # 显示χ²/ndf
            "show_yields": True,      # 显示拟合的yield值
            "extra_text": [           # 额外文本
                "Bin: 1.5 < p_{T} < 2.0 GeV/c",
                "Run period: 2021"
            ],
            "extra_entries": 2  # 额外条目数（用于计算图例大小）
        },
        
        "show_legend": True  # 是否显示图例
    }
)
```

### 完整示例：精美的绘图

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/phi_fit",
    variables=[("phi_M", 1.0, 1.06)],
    
    pdfs=[
        {"name": "sig", "var": "phi_M", "type": "bw_gauss",
         "config": {"mass": 1.0195, "width": 0.004249}},
        {"name": "bkg", "var": "phi_M", "type": "polynomial",
         "config": {"order": 1}}
    ],
    
    model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg",
    
    plot_config={
        "nbin": 60,
        
        "xlabel": {
            "phi_M": "M_{K^{+}K^{-}} (GeV/c^{2})"
        },
        
        "ylabel": {
            "phi_M": "Candidates / (1 MeV/c^{2})"
        },
        
        "components": {
            "sig": {
                "label": "Signal (#phi #rightarrow K^{+}K^{-})",
                "color": ROOT.kRed,
                "style": 4,  # Dash-dot line
                "width": 3
            },
            "bkg": {
                "label": "Combinatorial background",
                "color": ROOT.kGreen+2,
                "style": 7,  # Dotted line
                "width": 3
            }
        },
        
        "legend": {
            "x1": 0.60, "x2": 0.92,
            "y2": 0.88,
            "show_chi2": True,
            "show_yields": True,
            "extra_text": [
                "Belle II Preliminary",
                "#sqrt{s} = 10.58 GeV"
            ],
            "extra_entries": 2
        }
    }
)

result, fit_results = fit.run()
```

### 2D拟合的绘图配置

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/diphi",
    variables=[("phi1_M", 1.0, 1.06), ("phi2_M", 1.0, 1.06)],
    
    pdfs=[
        {"name": "sig1", "var": "phi1_M", "type": "bw_gauss", "config": {...}},
        {"name": "sig2", "var": "phi2_M", "type": "bw_gauss", "config": {...}},
        {"name": "bkg1", "var": "phi1_M", "type": "polynomial", "config": {"order": 1}},
        {"name": "bkg2", "var": "phi2_M", "type": "polynomial", "config": {"order": 1}}
    ],
    
    model=(
        "nsig[100,0,5000]*PROD(sig1, sig2) + "
        "nbkg1[50,0,2000]*PROD(bkg1, bkg2) + "
        "nbkg2[50,0,2000]*PROD(sig1, bkg2) + "
        "nbkg3[50,0,2000]*PROD(bkg1, sig2)"
    ),
    
    # 默认绘制所有变量
    plot_vars=["phi1_M", "phi2_M"],  # 可省略，会自动绘制所有变量
    
    plot_config={
        "nbin": 60,
        
        # 每个变量有自己的标签
        "xlabel": {
            "phi1_M": "M_{K^{+}K^{-}}^{(1)} (GeV/c^{2})",
            "phi2_M": "M_{K^{+}K^{-}}^{(2)} (GeV/c^{2})"
        },
        
        # 绘制的组件（投影到1D时会自动处理）
        "components": {
            "tmp_prod_1": {  # PROD(sig1, sig2)的自动生成名称
                "label": "Signal",
                "color": ROOT.kRed,
                "style": 4,
                "width": 3
            },
            "tmp_prod_2": {  # PROD(bkg1, bkg2)
                "label": "Background",
                "color": ROOT.kGreen+2,
                "style": 7,
                "width": 3
            }
        },
        
        "legend": {
            "show_chi2": True,
            "show_yields": True
        }
    }
)
```
### 最简配置

如果不需要精细控制，可以使用最简配置：

```python
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("phi_M", 1.0, 1.06)],
    pdfs=[...],
    model="...",
    
    # 最简配置：只指定要绘制的组件
    plot_config={
        "components": {
            "sig": {"label": "Signal"},
            "bkg": {"label": "Background"}
        }
    }
)
```

系统会使用默认值：
- xlabel/ylabel: 自动生成
- 颜色: sig用红色，bkg用绿色
- 线型: 默认样式
- 图例: 自动定位和大小

```python
from FIT.pdf_builders import PDFBuilder, PDF_REGISTRY

class MyCustomPDFBuilder(PDFBuilder):
    """自定义PDF构建器"""
    
    def build(self, workspace, var_name, config, pdf_name):
        """
        构建自定义PDF
        
        Args:
            workspace: RooWorkspace
            var_name: 变量名
            config: 配置字典
            pdf_name: PDF名称（必需参数）
        """
        # 从config中提取参数
        param1 = config.get("param1", 1.0)
        param2 = config.get("param2", 0.5)
        
        # 使用RooFit factory创建PDF
        workspace.factory(
            f"YourCustomPDF::{pdf_name}({var_name}, "
            f"param1[{param1}], param2[{param2}])"
        )

# 注册到registry（统一注册，不分signal/background）
PDF_REGISTRY.register("my_custom", MyCustomPDFBuilder())

# 使用自定义PDF
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("x", 0, 10)],
    pdfs=[{
        "name": "my_pdf",
        "var": "x",
        "type": "my_custom",  # 使用你的自定义PDF
        "config": {"param1": 1.5, "param2": 0.8}
    }],
    model="n[100,0,1000]*my_pdf"
)
```

**关键点**：
- 继承`PDFBuilder`基类
- 实现`build(workspace, var_name, config, pdf_name)`方法
- `pdf_name`是**必需参数**（v2设计改进）
- 使用`PDF_REGISTRY.register()`统一注册（不分signal/background）
    tree=tree,
    output_dir="output/",
    variables=[("x", 0, 10)],
    signal_pdfs=[{
        "var": "x",
        "type": "my_custom",  # 使用你的自定义PDF
        "config": {...}
    }],
    ...
)
```

## 可用的PDF类型

**重要**：v2设计中，所有PDF统一管理，不再区分Signal和Background！

### 常用PDFs

| 类型 | 用途 | 参数 |
|------|------|------|
| `bw_gauss` | Breit-Wigner ⊗ Gaussian | mass, width, sigma |
| `double_gauss_bw` | BW ⊗ 双高斯 | mass, width, sigma1, sigma2, frac |
| `crystal_ball` | Crystal Ball（非对称尾部） | mean, sigma, alpha, n |
| `voigtian` | Voigtian | mass, width, sigma |
| `gaussian` | 简单高斯 | mean, sigma |
| `polynomial` | 多项式 | order |
| `chebychev` | Chebychev多项式 | order |
| `argus` | ARGUS（运动学端点） | m0, c, p |
| `exponential` | 指数 | tau |
| `flat` | 平坦 | 无参数 |

### 查看可用PDF

```python
from FIT.pdf_builders import PDF_REGISTRY

# 列出所有注册的PDF类型
print(PDF_REGISTRY.list_types())
# 输出: ['bw_gauss', 'double_gauss_bw', 'crystal_ball', 'voigtian', 
#        'gaussian', 'polynomial', 'chebychev', 'argus', 'exponential', 'flat']
```

## 优势总结

### ✅ 代码架构简化
- **v1 (旧设计)**：专门类 + signal/background区分 → 复杂且不必要
- **v2 (新设计)**：单一GenericFit + 统一PDF + 灵活model解析 → 简洁强大

### ✅ 易于扩展
- 添加新PDF：创建Builder → 注册 → 直接使用
- 不需要修改GenericFit或其他代码

### ✅ 高度灵活
- 任意组合PDF（PROD, FCONV, 括号）
- 适合系统误差研究（轻松切换PDF类型）
- model字符串即文档

### ✅ 可维护性强
- 关注点分离：PDF构建 ↔ 拟合流程
- 高代码复用率
- 统一接口，易于理解

### ✅ 直观易用
- 配置即文档：一眼看出使用了什么PDF和结构
- model字符串直接表达物理意义
- 无需学习复杂的API

## 设计原则与模式

### SOLID原则

1. **单一职责 (Single Responsibility)**
   - `PDFBuilder`: 专注PDF构建
   - `ModelParser`: 专注model字符串解析
   - `GenericFit`: 专注拟合工作流
   
2. **开闭原则 (Open-Closed)**
   - 对扩展开放：添加新PDF Builder无需修改现有代码
   - 对修改封闭：GenericFit核心逻辑稳定

3. **依赖倒置 (Dependency Inversion)**
   - GenericFit依赖PDFBuilder抽象接口
   - 不依赖具体PDF实现细节

### 设计模式

- **Builder模式**：PDFBuilder封装PDF构建逻辑
- **Registry模式**：PDFBuilderRegistry管理所有builders
- **Strategy模式**：不同PDF类型可互换
- **Parser模式**：ModelParser解析复杂字符串

## 从旧代码迁移

### 旧方式（fit_functions.py）

```python
from FIT.fit_functions import perform_resonance_fit

result, nsig, nsig_err = perform_resonance_fit(
    tree, 
    output_dir,
    particle_config=("phi", 1.0195, 0.004249),
    fit_config=(1.0, 1.06, 60),
    which_bkg=1,
    bkg_order=1
)
```

### 新方式（GenericFit v2）

```python
from FIT.generic_fit import GenericFit

fit = GenericFit(
    tree=tree,
    output_dir=output_dir,
    variables=[("phi_M", 1.0, 1.06)],
    pdfs=[
        {
            "name": "sig",
            "var": "phi_M",
            "type": "bw_gauss",
            "config": {"mass": 1.0195, "width": 0.004249}
        },
        {
            "name": "bkg",
            "var": "phi_M",
            "type": "polynomial",
            "config": {"order": 1}
        }
    ],
    model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg"
)

result, fit_results = fit.run()
nsig = fit_results["nsig"]
nsig_err = fit_results["nsig_err"]
```

### 迁移对照表

| 旧参数 | 新参数 | 说明 |
|--------|--------|------|
| `particle_config` | `pdfs[i]["config"]` | PDF配置参数 |
| `fit_config` | `variables` | 拟合范围 |
| `which_bkg` | `pdfs[i]["type"]` | 选择PDF类型 |
| `bkg_order` | `config["order"]` | 多项式阶数 |
| 返回值 | `fit_results` | 字典形式，更灵活 |
## 核心思想总结

> **所有拟合本质上都是相同的流程，只是PDF配置和组合方式不同！**

### 关键洞察

1. **PDF无本质区别**
   - Signal和Background只是物理解释，数学上都是PDF
   - 统一处理所有PDF，在使用时赋予含义

2. **Model字符串即结构**
   - 直接表达物理模型：`nsig*sig + nbkg*bkg`
   - 支持复杂操作：`PROD`, `FCONV`, 括号
   - 无需额外配置文件

3. **分离关注点**
   - PDF构建：由Builder负责
   - Model解析：由Parser负责
   - 拟合流程：由GenericFit负责

### 设计演进历程

```
v0 (最初) → v1 (中间) → v2 (当前)
多个拟合类  单一类+分类PDF  单一类+统一PDF+解析器
```

- **v0**: 为每种场景写专门的函数/类 → 代码重复多
- **v1**: 统一GenericFit，但区分signal/background → 人为限制
- **v2**: 完全统一，model字符串表达一切 → **最优解**

---

## 快速开始

```python
from FIT.generic_fit import GenericFit

# 1. 准备数据
tree = ROOT.TFile("data.root").Get("tree")

# 2. 定义拟合
fit = GenericFit(
    tree=tree,
    output_dir="output/",
    variables=[("mass", 1.0, 1.06)],
    pdfs=[
        {"name": "sig", "var": "mass", "type": "bw_gauss", 
         "config": {"mass": 1.0195, "width": 0.004249}},
        {"name": "bkg", "var": "mass", "type": "polynomial", 
         "config": {"order": 1}}
    ],
    model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg"
)

# 3. 运行拟合
result, fit_results = fit.run()

# 4. 获取结果
print(f"Signal: {fit_results['nsig']:.0f} ± {fit_results['nsig_err']:.0f}")
print(f"Background: {fit_results['nbkg']:.0f} ± {fit_results['nbkg_err']:.0f}")
```

就这么简单！🎉

---

## 更多资源

- **完整示例**: `usage_examples_with_parser.py`
- **PDF Builders源码**: `pdf_builders.py`
- **Model解析器**: `model_parser.py`
- **GenericFit实现**: `generic_fit.py`
