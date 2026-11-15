import json
import unittest
from unittest.mock import patch

from scripts.evaluate_assistant import (
    DEFAULT_CASES_PATH,
    load_jsonl,
    query_endpoint,
    run_offline_pipeline_check,
    score_response,
    summarize_results,
    validate_eval_cases,
)
from training.train_lora import apply_preset, build_parser, parse_target_modules


class EvalSuiteTests(unittest.TestCase):
    def test_eval_cases_jsonl_schema_and_balance(self):
        cases = load_jsonl(DEFAULT_CASES_PATH)
        summary = validate_eval_cases(cases)
        self.assertGreaterEqual(summary['total_cases'], 35)
        self.assertEqual(
            summary['dimensions'],
            ['akurat', 'bersih', 'fokus', 'proporsional', 'tuntas'],
        )

    def test_scorer_flags_leaks_and_topic_drift(self):
        case = {
            'id': 'TEST-01',
            'dimension': 'bersih',
            'required_points': [['transfer learning']],
            'forbidden_terms': ['[s1]'],
            'disallowed_history_terms': ['flowchart'],
            'min_words': 10,
            'max_words': 80,
            'expect_limitation': False,
            'require_question': False,
            'allow_question': False,
        }
        good = score_response(
            case,
            'Transfer learning memakai ulang pengetahuan model yang sudah dilatih sebelumnya pada dataset besar.',
        )
        self.assertTrue(good['passed'])

        bad = score_response(
            case,
            '[S1] Transfer learning berbeda dari flowchart tadi. Mau lanjut?',
        )
        self.assertFalse(bad['passed'])
        self.assertLess(bad['scores']['bersih'], 1.0)
        self.assertLess(bad['scores']['fokus'], 1.0)
        self.assertLess(bad['scores']['proporsional'], 1.0)

    def test_scorer_requires_honest_limitation_when_expected(self):
        case = {
            'id': 'TEST-LIM-01',
            'dimension': 'akurat',
            'required_points': [],
            'forbidden_terms': ['zona'],
            'disallowed_history_terms': [],
            'min_words': 10,
            'max_words': 80,
            'expect_limitation': True,
            'require_question': False,
            'allow_question': False,
        }
        honest = score_response(
            case,
            'Istilah tersebut tidak ditemukan di materi KKA saat ini; silakan tanyakan kepada guru.',
        )
        self.assertTrue(honest['passed'])

        hallucinated = score_response(
            case,
            'Singkatan itu berarti Zona Belajar Cepat dalam kurikulum sekolah.',
        )
        self.assertFalse(hallucinated['passed'])
        self.assertEqual(hallucinated['scores']['akurat'], 0.0)

    def test_offline_check_and_summarize(self):
        res = run_offline_pipeline_check()
        self.assertIn('clean_score', res)
        summary = summarize_results([
            {'passed': True, 'scores': res['clean_score']},
            {'passed': False, 'scores': res['dirty_score']},
        ])
        self.assertEqual(summary['total'], 2)
        self.assertEqual(summary['passed'], 1)

    def test_train_lora_rtx3050_preset(self):
        args = build_parser().parse_args([
            '--data', 'data/lora/lora_train.jsonl',
            '--out', 'adapters/test',
            '--preset', 'rtx3050-6gb',
        ])
        apply_preset(args)
        self.assertEqual(args.precision, 'fp16')
        self.assertEqual(args.batch_size, 1)
        self.assertGreaterEqual(args.grad_accum, 8)
        self.assertLessEqual(args.max_length, 1536)
        self.assertEqual(args.lora_r, 8)
        self.assertEqual(args.lora_alpha, 16)

    def test_query_endpoint_matches_chat_contract(self):
        captured = []

        class FakeResp:
            def __init__(self, body):
                self._raw = json.dumps(body).encode('utf-8')

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self._raw

        def fake_urlopen(req, timeout=90):
            captured.append((req.get_header('X-api-key'), json.loads(req.data.decode('utf-8'))))
            return FakeResp({'session_id': 'eval-S1', 'answer': 'Jawaban bersih.'})

        case = {
            'id': 'S1',
            'mode': 'umum',
            'question': 'Apa itu Transfer Learning?',
            'history': [{'role': 'user', 'content': 'Apa itu flowchart?'}],
        }
        with patch('scripts.evaluate_assistant.urllib.request.urlopen', side_effect=fake_urlopen):
            ans = query_endpoint('http://127.0.0.1:8000/api/chat', case, api_key='secret-key')
        self.assertEqual(ans, 'Jawaban bersih.')
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0][0], 'secret-key')
        self.assertIsNone(captured[0][1]['session_id'])
        self.assertTrue(captured[0][1]['reset_context'])
        self.assertEqual(captured[1][1]['session_id'], 'eval-S1')
        self.assertEqual(captured[1][1]['question'], 'Apa itu Transfer Learning?')
        self.assertFalse(captured[1][1]['reset_context'])

    def test_train_lora_regularization_flags(self):
        parser = build_parser()
        args = apply_preset(parser.parse_args([
            '--data', 'd.jsonl',
            '--out', 'o',
            '--preset', 'rtx3050-6gb',
            '--precision', 'bf16',
            '--neftune-noise-alpha', '5.0',
            '--max-grad-norm', '0.5',
            '--target-modules', 'q_proj,k_proj,v_proj,o_proj',
        ]))
        self.assertEqual(args.precision, 'bf16')
        self.assertEqual(args.max_length, 1536)
        self.assertEqual(args.lora_r, 8)
        self.assertEqual(args.lora_alpha, 16)
        self.assertEqual(args.neftune_noise_alpha, 5.0)
        self.assertEqual(args.max_grad_norm, 0.5)
        self.assertEqual(args.empty_cache_steps, 4)
        self.assertEqual(
            parse_target_modules(args.target_modules),
            ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        )
        self.assertEqual(parse_target_modules('all-linear'), 'all-linear')

        bad_neftune = parser.parse_args([
            '--data', 'd.jsonl', '--out', 'o', '--neftune-noise-alpha', '-0.1',
        ])
        with self.assertRaises(ValueError):
            apply_preset(bad_neftune)

        bad_clip = parser.parse_args([
            '--data', 'd.jsonl', '--out', 'o', '--max-grad-norm', '-1',
        ])
        with self.assertRaises(ValueError):
            apply_preset(bad_clip)

        bad_cache = parser.parse_args([
            '--data', 'd.jsonl', '--out', 'o', '--empty-cache-steps', '-2',
        ])
        with self.assertRaises(ValueError):
            apply_preset(bad_cache)

        bad_modules = parser.parse_args([
            '--data', 'd.jsonl', '--out', 'o', '--target-modules', ' , ',
        ])
        with self.assertRaises(ValueError):
            apply_preset(bad_modules)


if __name__ == '__main__':
    unittest.main()
