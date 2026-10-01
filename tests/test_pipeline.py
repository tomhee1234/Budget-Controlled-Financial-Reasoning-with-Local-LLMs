import contextlib
import io
import json
from pathlib import Path
import random
import re
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

import conditions
import config
from llm_client import LlamaCppClient
import metrics
import run_experiment
import task_generator


def response(text, tokens=10, stop='word'):
    return {'text': text, 'tokens_predicted': tokens, 'time_seconds': 0.1,
            'stopped_limit': stop == 'limit', 'stopped_word': stop == 'word', 'stop_type': stop}


class ConditionsTests(unittest.TestCase):
    def test_forced_suffix_is_parsed_and_think_block_closed(self):
        for function in (conditions.run_condition_c, conditions.run_condition_d, conditions.run_condition_e):
            client = Mock()
            client.complete.side_effect = [response('Calculation', 256, 'limit'), response(' 0.226')]
            result = function(client, 'Question', 256)
            self.assertEqual(metrics.extract_numeric_answer(result.final_text), 0.226)
            prompt = client.complete.call_args.args[0]
            self.assertEqual(prompt.count('</think>'), 1)
            self.assertTrue(prompt.endswith('</think>\n\nFinal Answer:'))
            self.assertEqual(result.total_tokens, 266)
            self.assertEqual(client.complete.call_args.kwargs['n_predict'], 200)

    def test_adaptive_extends_only_on_limit_and_warning(self):
        client = Mock()
        client.complete.side_effect = [response('Wait, reconsider', 256, 'limit'), response('Checked', 12), response(' 1')]
        result = conditions.run_condition_f(client, 'Question')
        self.assertEqual(result.extensions_used, 1)
        self.assertIn('wait', result.extension_triggers)
        self.assertEqual(result.reasoning_tokens, 268)
        self.assertEqual(client.complete.call_count, 3)
        client = Mock()
        client.complete.side_effect = [response('Wait, reconsider', 12), response(' 1')]
        result = conditions.run_condition_f(client, 'Question')
        self.assertEqual(result.extensions_used, 0)

    def test_free_reasoning_never_scores_an_answer_inside_unclosed_think(self):
        client = Mock()
        client.complete.return_value = response('Maybe Final Answer: 123', 1024, 'limit')
        result = conditions.run_condition_b(client, 'Question')
        self.assertIsNone(metrics.extract_numeric_answer(result.final_text))
        self.assertEqual(result.reasoning_tokens, 1024)
        self.assertTrue(result.safety_limit_reached)
        client.complete.return_value = response('Thinking</think>Final Answer: 0.5', 20)
        client.count_tokens.return_value = 8
        result = conditions.run_condition_b(client, 'Question')
        self.assertEqual(result.reasoning_tokens, 8)
        self.assertEqual(result.total_tokens, 20)
        self.assertEqual(metrics.extract_numeric_answer(result.final_text), 0.5)


class ClientTests(unittest.TestCase):
    @patch('llm_client.requests.post')
    def test_new_and_old_stop_fields_and_seed(self, post):
        client = LlamaCppClient('http://test')
        client.set_seed(100)
        for data, limit in [({'stop_type': 'limit'}, True), ({'stop_type': 'word'}, False),
                            ({'stopped_limit': True}, True), ({'stopped_word': True}, False)]:
            post.return_value.json.return_value = {'content': 'abc', 'tokens_predicted': 3, **data}
            self.assertEqual(client.complete('test', 3)['stopped_limit'], limit)
        self.assertEqual(post.call_args.kwargs['json']['seed'], 103)
        self.assertEqual(post.call_args.kwargs['json']['min_p'], 0)

    @patch('llm_client.requests.post')
    def test_rejects_truncation_and_unknown_schema(self, post):
        client = LlamaCppClient('http://test')
        for data in [{}, {'content': 'a', 'tokens_predicted': 1},
                     {'content': 'a', 'tokens_predicted': 1, 'stop_type': 'limit', 'truncated': True}]:
            post.return_value.json.return_value = data
            with self.assertRaises(ValueError):
                client.complete('test', 1)


class MetricsTests(unittest.TestCase):
    def test_numeric_formats(self):
        for text, expected in [('Final Answer: $1,234.56', 1234.56),
                               ('Final Answer: -.25', -.25), ('Final Answer: +2e-3', .002),
                               ('Final Answer: 1\nFinal Answer: 2', 2),
                               ('### Final Answer:\n\n$$\n\\boxed{0.303}\n$$', .303),
                               (r'$$\text{Final Answer: } -0.3091$$', -.3091),
                               ('**Final Answer:** **−0.25**', -.25)]:
            self.assertEqual(metrics.extract_numeric_answer(text), expected)
        for text in ['reasoning 123', 'Final Answer: 1e9999', 'Final Answer: 12,34',
                     r'Calculation: \boxed{123}', r'Final Answer: \frac{1}{2}']:
            self.assertIsNone(metrics.extract_numeric_answer(text))

    def test_ground_truth_uses_displayed_inputs(self):
        rng = random.Random(5)
        for _ in range(100):
            task = task_generator.gen_correlation_from_covariance(rng)
            values = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', task['text'])]
            self.assertAlmostEqual(task['ground_truth'], values[2] / (values[0] / 100 * values[1] / 100), places=10)
            task = task_generator.gen_multi_step_sharpe_from_prices(rng)
            values = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', task['text'])]
            self.assertAlmostEqual(task['ground_truth'], ((values[1] - values[0]) / values[0] - values[3] / 100) / (values[2] / 100), places=10)


class RunnerTests(unittest.TestCase):
    def test_fixed_conditions_share_pre_intervention_seed(self):
        seeds = [run_experiment.seed_for('task', 0, c) for c in
                 ('C_fixed_budget512', 'D_generic_budget512', 'E_financial_budget512')]
        self.assertEqual(len(set(seeds)), 1)
        self.assertNotEqual(seeds[0], run_experiment.seed_for('task', 1, 'C_fixed_budget512'))

    def test_selected_conditions_only_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temporary)
            tasks = folder / 'tasks.json'
            tasks.write_text(json.dumps(task_generator.generate_dataset(1)))
            run = folder / 'run'
            run_experiment.run_all(tasks, True, run_dir=run, repetitions=1, selected_conditions=['B_free'])
            records = list(run.glob('*/records/*.json'))
            self.assertEqual(len(records), 8)
            self.assertTrue(all(json.loads(p.read_text())['condition'] == 'B_free' for p in records))
            with self.assertRaises(ValueError):
                run_experiment.run_all(tasks, True, run_dir=run, repetitions=1, resume=True)

    def test_cancellation_prevents_new_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            cancel = threading.Event()
            cancel.set()
            with patch('run_experiment.get_client') as client:
                run_experiment.worker(config.QUANTIZATIONS[0], task_generator.generate_dataset(1),
                                      1, Path(temporary) / 'worker', True, cancel)
                client.return_value.complete.assert_not_called()

    def test_parallel_mock_resume_and_changed_settings_guard(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temporary)
            tasks = folder / 'tasks.json'
            tasks.write_text(json.dumps(task_generator.generate_dataset(1)))
            run = folder / 'run'
            run_experiment.run_all(tasks, True, run_dir=run, repetitions=1)
            records = list(run.glob('*/records/*.json'))
            self.assertEqual(len(records), 96)
            before = {str(p): p.stat().st_mtime_ns for p in records}
            run_experiment.run_all(tasks, True, run_dir=run, repetitions=1, resume=True)
            self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in records})
            with patch.object(config, 'TEMPERATURE', .5), self.assertRaises(ValueError):
                run_experiment.run_all(tasks, True, run_dir=run, repetitions=1, resume=True)
            with self.assertRaises(FileExistsError):
                run_experiment.run_all(tasks, True, run_dir=run, repetitions=1)

    def test_partial_failure_keeps_completed_condition(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'worker'
            quant = config.QUANTIZATIONS[0]
            task = task_generator.generate_dataset(1)[0]
            def fail(*args):
                raise RuntimeError('connection lost')
            specs = [('A_direct', conditions.run_condition_a, ()), ('B_free', fail, ())]
            with patch('run_experiment.condition_specs', return_value=specs), self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                run_experiment.worker(quant, [task], 1, folder, True)
            self.assertEqual(len(list((folder / 'records').glob('*.json'))), 1)
            self.assertTrue((folder / 'experiment.csv').exists())


if __name__ == '__main__':
    unittest.main()
