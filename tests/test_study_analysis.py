import tempfile
import unittest
from pathlib import Path
import pandas as pd
import study_analysis


class StudyAnalysisTests(unittest.TestCase):
    def test_repetitions_do_not_become_independent_tasks(self):
        rows = []
        for task in range(6):
            for rep in range(2):
                for condition, correct in [('D_generic_budget512', False), ('E_financial_budget512', True)]:
                    rows.append(dict(quantization='Q4_K_M', task_id=str(task), repetition=rep,
                                     condition=condition, correct=correct, predicted_answer=1,
                                     total_tokens=100, total_time_seconds=1, safety_limit_reached=False))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            study_analysis.analyze([path], folder)
            comparisons = pd.read_csv(Path(folder) / 'paired_task_comparisons.csv')
            self.assertEqual(comparisons.iloc[0]['n_tasks'], 6)
            self.assertEqual(comparisons.iloc[0]['n_pairs'], 12)
            self.assertEqual(comparisons.iloc[0]['accuracy_difference'], 1)
            self.assertEqual(comparisons.iloc[0]['ci95_low'], 1)
            with self.assertRaises(ValueError):
                study_analysis.analyze([path, path], folder)

    def test_no_interval_claim_for_tiny_pilot(self):
        self.assertEqual(study_analysis.mean_ci([0, 1, 0, 1]), (None, None))
