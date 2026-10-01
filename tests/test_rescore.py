import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rescore_run import rescore


class RescoreTests(unittest.TestCase):
    def test_rescoring_preserves_original_and_uses_recorded_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'original'
            records = run / 'Q4_K_M' / 'records'
            records.mkdir(parents=True)
            (run / 'manifest.json').write_text(json.dumps({
                'expected_results': 1, 'identity': {'settings': {'TOLERANCE': 0.00001}}}), encoding='utf-8')
            (run / 'status.json').write_text('{"state": "complete"}', encoding='utf-8')
            row = {'quantization': 'Q4_K_M', 'task_id': 'correlation', 'repetition': 0,
                   'condition': 'B_free', 'predicted_answer': None, 'correct': False,
                   'error_category_heuristic': 'keine_antwort_extrahiert',
                   'ground_truth': -0.30907, 'final_text': r'\text{Final Answer: } -0.3091'}
            original = records / 'record.json'
            original.write_text(json.dumps(row), encoding='utf-8')
            before = original.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                rescore(run, root / 'rescored')
            self.assertEqual(original.read_bytes(), before)
            audit = json.loads((root / 'rescored' / 'rescoring_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['changes'][0]['after']['predicted_answer'], -0.3091)
            self.assertFalse(audit['changes'][0]['after']['correct'])
            self.assertTrue((root / 'rescored' / 'Q4_K_M' / 'experiment.csv').is_file())
            with self.assertRaises(FileExistsError):
                rescore(run, root / 'rescored')
