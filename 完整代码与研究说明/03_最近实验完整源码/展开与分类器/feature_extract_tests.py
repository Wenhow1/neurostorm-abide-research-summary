"""CPU-only checks of feature ordering, input partitions, receipts, and thermal guards.

No checkpoint is opened and no model inference or training is performed here.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('feature_extract', HERE / 'feature_extract.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def telemetry(temperature, fraction=.1):
    return {'physical_index': 1, 'uuid': f.GPU_UUID, 'temperature_C': temperature,
            'memory_fraction': fraction, 'memory_used_MiB': 1000 * fraction,
            'memory_total_MiB': 1000., 'time_unix': 0.}


class FeatureTests(unittest.TestCase):
    def test_flatten_is_roi_then_channel_and_reversible(self):
        for d in [288, 576]:
            z = np.arange(2 * 116 * d, dtype=np.float32).reshape(2, 1, 116, d)
            flat = f.flatten_tokens(z, d, np)
            self.assertEqual(flat.shape, (2, 116 * d))
            self.assertTrue(np.array_equal(flat.reshape(2, 116, d), z[:, 0]))
            self.assertEqual(flat[0, d], z[0, 0, 1, 0])
            self.assertTrue(flat.flags.c_contiguous)

    def test_wrong_shape_and_nan_rejected(self):
        with self.assertRaises(AssertionError): f.flatten_tokens(np.zeros((2, 116, 288)), 288, np)
        z = np.zeros((1, 1, 116, 288)); z[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): f.flatten_tokens(z, 288, np)

    def test_train_downstream_disjointness(self):
        raw = np.zeros((290, 116, 116), dtype=np.float32)
        ids = np.arange(1000, 1290)
        norm = {'mean': 0., 'std': 1., 'fit_subject_ids': list(range(610))}
        f.validate_source(raw, ids, norm, np)
        bad = dict(norm); bad['fit_subject_ids'] = list(range(609)) + [1000]
        with self.assertRaises(AssertionError): f.validate_source(raw, ids, bad, np)
        bad = dict(norm); bad['std'] = 0.
        with self.assertRaises(AssertionError): f.validate_source(raw, ids, bad, np)

    def test_exact_complete_receipt_resume_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root / 'x.npy'; f.atomic_npy(p, np.ones((2, 3)), np)
            rec = root / 'receipt.json'
            f.put(rec, {'status': 'COMPLETE', 'signature': 'same',
                        'records': [{'path': 'x.npy', 'feature_sha256': f.sha(p)}]})
            self.assertEqual(f.check_resume(rec, 'same', root)['status'], 'COMPLETE')
            with self.assertRaises(RuntimeError): f.check_resume(rec, 'other', root)
            p.write_bytes(b'tampered')
            with self.assertRaises(RuntimeError): f.check_resume(rec, 'same', root)

    def guard(self, root, rows, limit=7200):
        values = iter(rows); now = [0.]
        def sleep(seconds): now[0] += seconds
        return f.ThermalGuard(Path(root) / 'status.json', Path(root) / 'events.jsonl', {'name': 'test'},
                              query=lambda: next(values), sleeper=sleep, clock=lambda: now[0], wait_limit=limit)

    def test_temperature_hysteresis_requires_65(self):
        with tempfile.TemporaryDirectory() as temp:
            guard = self.guard(temp, [telemetry(80), telemetry(70), telemetry(65)])
            last = guard.check(completed_subjects=0)
            self.assertEqual(last['temperature_C'], 65)
            self.assertEqual(guard.wait_seconds, 10)
            events = [json.loads(s)['event'] for s in (Path(temp) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events, ['PAUSE', 'RESUME'])

    def test_memory_pause_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            guard = self.guard(temp, [telemetry(50, .9), telemetry(50, .95), telemetry(50, .89)])
            self.assertLess(guard.check()['memory_fraction'], .9)
            self.assertEqual(guard.wait_seconds, 10)

    def test_wait_timeout_and_telemetry_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            guard = self.guard(temp, [telemetry(81), telemetry(81)], limit=5)
            with self.assertRaises(TimeoutError): guard.check()
            guard = self.guard(temp, [])
            with self.assertRaises(StopIteration): guard.check()


if __name__ == '__main__': unittest.main(verbosity=2)
