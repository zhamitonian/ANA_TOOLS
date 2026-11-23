import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "b1MCGen.py"
spec = importlib.util.spec_from_file_location("b1mcgen", SCRIPT)
b1mcgen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b1mcgen
spec.loader.exec_module(b1mcgen)
runner_spec = importlib.util.spec_from_file_location("gsim_runner", SCRIPT.with_name("b1MCGen_gsim_runner.py"))
gsim_runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(gsim_runner)


class GsimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.gen = self.work / "gen"
        self.gen.mkdir()
        self.mc = b1mcgen.Belle1MCGen.__new__(b1mcgen.Belle1MCGen)
        self.mc.work_dir = str(self.work)
        self.mc.selected_runs = [
            b1mcgen.RunDatabaseEntry(55, 3, background_file="/tmp/background"),
            b1mcgen.RunDatabaseEntry(19, 2, background_file="/tmp/background"),
        ]
        self.mc.queue = "s"
        self.mc.dry_run = True
        self.mc.job_ids = []

    def input(self, relative):
        path = self.gen / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_distinct_inputs_and_bellemcoutput_configuration(self):
        inputs = ["e55_r3_j0.dat", "e55_r3_j1.dat",
                  "a/e55_r3_j0.dat", "b/e55_r3_j0.dat", "e19_r2_j0.dat"]
        for name in inputs:
            self.input(name)
        with mock.patch("subprocess.run") as run:
            self.mc.submit_gsim_jobs()
            run.assert_not_called()
        for name in inputs:
            stem = Path(name).with_suffix("")
            basf = self.work / "sim_script" / (str(stem) + "_gsim.basf")
            card = self.work / "sim_script" / (str(stem) + "_gsim.dat")
            wrapper = self.work / "sim_script" / (str(stem) + "_gsim_run.sh")
            mdst = self.work / "sim" / (str(stem) + ".mdst")
            data = basf.read_bytes()
            self.assertNotIn(b"\0", data)
            self.assertNotIn(b"bpsmear", data)
            self.assertIn(f"GSIM_INPUT\\{card.name}\n".encode(), data)
            self.assertIn(f"output open {mdst}.partial".encode(), data)
            self.assertIn(f"process_event {self.gen / name}".encode(), data)
            self.assertIn(b"module put_parameter evtcls classification_level\\0\n", data)
            subprocess.run(["bash", "-n", str(wrapper)], check=True)
            submit = self.work / "sim_script" / (str(stem) + "_gsim_submit.sh")
            subprocess.run(["bash", "-n", str(submit)], check=True)
        with self.assertRaises(FileExistsError):
            self.mc.submit_gsim_jobs()

    def test_preexisting_output_prevents_all_script_creation_and_submission(self):
        self.input("e55_r3_j0.dat")
        self.input("e55_r3_j1.dat")
        sim = self.work / "sim"
        sim.mkdir()
        existing = sim / "e55_r3_j1.mdst"
        existing.write_text("existing output")
        self.mc.dry_run = False
        with mock.patch("subprocess.run") as run:
            with self.assertRaises(FileExistsError):
                self.mc.submit_gsim_jobs()
            run.assert_not_called()
        self.assertEqual(existing.read_text(), "existing output")
        self.assertEqual(list((self.work / "sim_script").iterdir()), [])

    def test_existing_index_and_lock_are_protected(self):
        self.input("e55_r3.dat")
        sim = self.work / "sim"
        sim.mkdir()
        for suffix in (".index", ".lock"):
            existing = sim / ("e55_r3.mdst" + suffix)
            existing.touch()
            with self.assertRaises(FileExistsError):
                self.mc.submit_gsim_jobs()
            existing.unlink()

    def test_rejects_whitespace_in_basf_paths(self):
        self.input("with space/e55_r3.dat")
        with self.assertRaisesRegex(ValueError, "whitespace"):
            self.mc.submit_gsim_jobs()

    def test_rejects_gsim_parameter_filename_over_limit(self):
        self.input("e55_r3_" + "a" * 123 + ".dat")
        with self.assertRaisesRegex(ValueError, "132-byte limit"):
            self.mc.submit_gsim_jobs()

    def test_readback_requires_one_matching_nonempty_run(self):
        log = "   Exp   Run    #evts        Lum\n    7     6     68     1.0\n"
        self.assertEqual(gsim_runner.event_count(log, 7, 6), 68)
        with self.assertRaises(ValueError):
            gsim_runner.event_count(log, 7, 7)
        with self.assertRaises(ValueError):
            gsim_runner.event_count(log + "  7  7  1  1.0\n", 7, 6)
        with self.assertRaises(ValueError):
            gsim_runner.event_count(log.replace("68", "0"), 7, 6)
        self.assertTrue(gsim_runner.failed_worker("event_process terminated; id=0, status=6"))
        self.assertFalse(gsim_runner.failed_worker("event_process terminated; id=0, status=0"))

    def test_luminosity_distribution_and_large_run_splitting(self):
        self.mc.selected_runs = [
            b1mcgen.RunDatabaseEntry(55, 3, luminosity=1, background_file="/bg"),
            b1mcgen.RunDatabaseEntry(55, 4, luminosity=1, background_file="/bg"),
            b1mcgen.RunDatabaseEntry(55, 5, luminosity=1, background_file="/bg"),
        ]
        self.mc.total_events = 2
        self.assertEqual(self.mc.distribute_events(), 2)
        self.assertEqual([r.events for r in self.mc.selected_runs], [1, 1, 0])
        self.mc.selected_runs = [b1mcgen.RunDatabaseEntry(55, 3, events=20001, background_file="/bg")]
        self.mc.model_file = str(self.work / "model.py")
        Path(self.mc.model_file).touch()
        self.mc.dry_run = False
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "Job <123>")) as run:
            self.mc.submit_generation_jobs()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([int(cmd[-3]) for cmd in commands], [9999, 9999, 3])
        self.assertEqual(len({cmd[-1] for cmd in commands}), 3)

    def test_incomplete_mdst_is_not_published(self):
        source = self.input("e55_r3.dat")
        output = self.work / "result.mdst"
        card = self.work / "gsim.dat"
        basf = self.work / "gsim.basf"
        wrapper = self.work / "gsim.sh"
        for path in (card, basf, wrapper):
            path.touch()

        def fake_gsim(*args, **kwargs):
            Path(str(output) + ".partial").write_bytes(b"partial output")
            return subprocess.CompletedProcess(args[0], 0, "GSIM completed\n")

        args = ["55", "3", str(source), str(output), str(card), str(basf), str(wrapper),
                str(self.work / "input.log"), str(self.work / "output.log")]
        with mock.patch.object(gsim_runner, "read_events", side_effect=[5, 1]), \
                mock.patch.object(gsim_runner.subprocess, "run", side_effect=fake_gsim):
            with self.assertRaisesRegex(RuntimeError, "Event count mismatch"):
                gsim_runner.main(args)
        self.assertFalse(output.exists())
        self.assertTrue(Path(str(output) + ".partial").exists())

    def test_event_loss_tolerance_and_worker_failure(self):
        source = self.input("e55_r3.dat")
        card = self.work / "gsim.dat"
        basf = self.work / "gsim.basf"
        wrapper = self.work / "gsim.sh"
        for path in (card, basf, wrapper):
            path.touch()
        cases = [
            ("equal", 10, False, 0, True),
            ("one", 9, True, 0, True),
            ("three", 7, True, 0, True),
            ("without_worker_failure", 8, False, 0, True),
            ("four", 6, True, 0, False),
            ("surplus", 11, False, 0, False),
            ("worker_with_equal_counts", 10, True, 0, False),
            ("nonzero_exit", 9, True, 1, False),
        ]
        for name, observed, worker_failed, returncode, succeeds in cases:
            with self.subTest(name=name):
                output = self.work / f"{name}.mdst"
                args = ["55", "3", str(source), str(output), str(card), str(basf), str(wrapper),
                        str(self.work / f"{name}_input.log"), str(self.work / f"{name}_output.log")]
                log = "event_process terminated; id=0, status=6\n" if worker_failed else "GSIM completed\n"

                def fake_gsim(*args, **kwargs):
                    Path(str(output) + ".partial").write_bytes(b"partial output")
                    return subprocess.CompletedProcess(args[0], returncode, log)

                stderr = io.StringIO()
                with mock.patch.object(gsim_runner, "read_events", side_effect=[10, observed]), \
                        mock.patch.object(gsim_runner.subprocess, "run", side_effect=fake_gsim), \
                        mock.patch("sys.stderr", stderr):
                    if succeeds:
                        gsim_runner.main(args)
                    else:
                        with self.assertRaises(RuntimeError):
                            gsim_runner.main(args)
                self.assertEqual(output.exists(), succeeds)
                if succeeds and observed != 10:
                    self.assertIn(f"expected=10, observed={observed}, missing={10 - observed}", stderr.getvalue())
                    self.assertIn(f"worker_failed={worker_failed}", stderr.getvalue())
                elif succeeds:
                    self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
