#!/usr/bin/env python3
"""Run and verify one Belle GSIM job without publishing incomplete output."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def event_count(log, experiment, run):
    rows = re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\S+", log, re.MULTILINE)
    if len(rows) != 1 or tuple(map(int, rows[0][:2])) != (experiment, run):
        raise ValueError(f"Expected one exp {experiment}, run {run} in BASF readback, found {rows}")
    count = int(rows[0][2])
    if count <= 0:
        raise ValueError("BASF readback contains no events")
    return count


def failed_worker(log):
    return bool(re.search(r"event_process terminated[^\n]*status=(?!0(?:\D|$))\d+", log))


def read_events(path, log_path, experiment, run):
    commands = f"path add_module main ProcessHeader\ninitialize\nprocess_event {path}\nterminate\n"
    setup = (
        "export BELLE_LEVEL=b20090127_0910 BELLE_DEBUG=opt CERN_LEVEL=2006 "
        "USE_GRAND_REPROCESS_DATA=1 BASF_NPROCESS=0; "
        "source /sw/belle/local/etc/bashrc_general || exit 1; exec basf"
    )
    result = subprocess.run(["bash", "-c", setup], input=commands, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(log_path, "x") as f:
        f.write(result.stdout)
    if result.returncode or failed_worker(result.stdout):
        raise RuntimeError(f"BASF readback failed for {path}; see {log_path}")
    return event_count(result.stdout, experiment, run)


def main(args):
    if len(args) != 9:
        raise ValueError("Expected exp, run, input, output, card, BASF script, wrapper and two readback logs")
    exp, run = int(args[0]), int(args[1])
    source, output, card, basf_script, wrapper, input_log, output_log = map(Path, args[2:])
    partial = Path(str(output) + ".partial")
    os.mkdir(str(output) + ".lock")
    for path in (output, Path(str(output) + ".index"), partial, Path(str(partial) + ".index")):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite {path}")
    expected = read_events(source, input_log, exp, run)

    with tempfile.TemporaryDirectory(prefix="b1gsim-", dir="/tmp") as tmp:
        scratch = Path(tmp)
        if exp <= 27:
            local_input = scratch / "input.dat"
            local_output = scratch / "output.mdst"
            local_card = scratch / card.name
            local_basf = scratch / basf_script.name
            local_wrapper = scratch / wrapper.name
            shutil.copyfile(source, local_input)
            if Path(str(source) + ".index").exists():
                shutil.copyfile(str(source) + ".index", str(local_input) + ".index")
            shutil.copyfile(card, local_card)
            text = basf_script.read_text()
            old_input = f"process_event {source}\n"
            old_output = f"output open {partial}\n"
            if text.count(old_input) != 1 or text.count(old_output) != 1:
                raise ValueError("GSIM script paths do not match the requested input and output")
            text = text.replace(old_input, f"process_event {local_input}\n")
            text = text.replace(old_output, f"output open {local_output}\n")
            local_basf.write_text(text)
            text = wrapper.read_text().replace(str(basf_script), str(local_basf))
            text = text.replace(str(wrapper.parent), str(scratch))
            text = text.replace(str(partial), str(local_output))
            local_wrapper.write_text(text)
            command = ["/sw/belle/local/bin/centos7-exec", "bash", str(local_wrapper)]
        else:
            local_output = partial
            command = ["bash", str(wrapper)]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout, end="", flush=True)
        if result.returncode or failed_worker(result.stdout):
            raise RuntimeError(f"GSIM failed (exit {result.returncode} or event worker failure)")
        if not local_output.is_file() or local_output.stat().st_size == 0:
            raise RuntimeError("GSIM did not produce a nonempty MDST")
        observed = read_events(local_output, output_log, exp, run)
        if observed != expected:
            raise RuntimeError(f"Event count mismatch: input {expected}, MDST {observed}")
        if exp <= 27:
            with local_output.open("rb") as src, partial.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            local_index = Path(str(local_output) + ".index")
            if local_index.exists():
                with local_index.open("rb") as src, open(str(partial) + ".index", "xb") as dst:
                    shutil.copyfileobj(src, dst)
        os.link(partial, output)
        if Path(str(partial) + ".index").exists():
            os.link(str(partial) + ".index", str(output) + ".index")
            os.unlink(str(partial) + ".index")
        os.unlink(partial)
        print(f"Verified exp {exp}, run {run}: {observed} events; published {output}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
