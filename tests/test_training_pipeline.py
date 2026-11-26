"""Unit tests for RTX 5090 SFT preset, DPO alignment pipeline, dataset auditor, and bias/grounding evals."""

import json
from pathlib import Path
import unittest

from scripts.audit_dataset_quality import audit_sft_split, run_full_audit
from scripts.evaluate_assistant import load_jsonl, validate_eval_cases
from training.train_dpo import (
    build_dpo_example,
    build_parser as build_dpo_parser,
    split_dpo_groups,
    validate_dpo_rows,
)
from training.train_lora import (
    apply_preset,
    build_parser as build_sft_parser,
    resolve_optimizer,
)

ROOT = Path(__file__).resolve().parent.parent


class TrainingPipelineTests(unittest.TestCase):
    def test_rtx5090_preset_configures_native_bf16_rslora_and_fused_adamw(self):
        args = build_sft_parser().parse_args([
            '--data', 'data/lora/lora_train.jsonl',
            '--out', 'adapters/test-run',
            '--preset', 'rtx5090-32gb',
        ])
        resolved = apply_preset(args)
        self.assertEqual(resolved.precision, 'bf16')
        self.assertTrue(resolved.no_4bit)
        self.assertTrue(resolved.use_rslora)
        self.assertTrue(resolved.group_by_length)
        self.assertEqual(resolved.neftune_noise_alpha, 5.0)
        self.assertEqual(resolve_optimizer(resolved, load_in_4bit=not resolved.no_4bit), 'adamw_torch_fused')

    def test_rtx3050_preset_keeps_4bit_memory_safeguards(self):
        args = build_sft_parser().parse_args([
            '--data', 'data/lora/lora_train.jsonl',
            '--out', 'adapters/test-run',
            '--preset', 'rtx3050-6gb',
        ])
        resolved = apply_preset(args)
        self.assertFalse(resolved.no_4bit)
        self.assertEqual(resolved.batch_size, 1)
        self.assertGreaterEqual(resolved.grad_accum, 8)
        self.assertLessEqual(resolved.max_length, 1536)
        self.assertLessEqual(resolved.lora_r, 8)

    def test_dpo_preferences_dataset_is_valid_and_converts_to_trl_format(self):
        dpo_path = ROOT / 'data' / 'lora' / 'dpo_preferences.jsonl'
        rows = [json.loads(line) for line in dpo_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 20)
        validate_dpo_rows(rows)
        train_rows, val_rows = split_dpo_groups(rows)
        self.assertFalse({r['group_id'] for r in train_rows} & {r['group_id'] for r in val_rows})

        sample = build_dpo_example(rows[0])
        self.assertEqual(sample['prompt'][0]['role'], 'system')
        self.assertEqual(sample['prompt'][-1]['role'], 'user')
        self.assertEqual(sample['chosen'][0]['role'], 'assistant')
        self.assertEqual(sample['rejected'][0]['role'], 'assistant')

        dpo_args = build_dpo_parser().parse_args(['--out', 'adapters/test-dpo'])
        self.assertEqual(dpo_args.loss_type, 'robust')
        self.assertEqual(dpo_args.lr, 5e-6)

    def test_dpo_validator_rejects_leaked_chosen_or_identical_pairs(self):
        bad_leak = [{
            'id': 'BAD-1', 'group_id': 'G1', 'category': 'clean', 'mode': 'umum',
            'context': '', 'question': 'Apa itu AI?',
            'chosen': 'AI adalah kecerdasan buatan [S1].',
            'rejected': 'AI adalah makanan.',
        }]
        with self.assertRaisesRegex(ValueError, 'artefak'):
            validate_dpo_rows(bad_leak)

        bad_dup = [{
            'id': 'BAD-2', 'group_id': 'G1', 'category': 'clean', 'mode': 'umum',
            'context': '', 'question': 'Apa itu AI?',
            'chosen': 'Sama persis.', 'rejected': 'Sama persis.',
        }]
        with self.assertRaisesRegex(ValueError, 'identik'):
            validate_dpo_rows(bad_dup)

    def test_dataset_auditor_flags_conflicts_and_real_dataset_has_no_leakage(self):
        synthetic = [{
            'id': 'SYN-1', 'group_id': 'G1', 'mode': 'umum',
            'context': 'Materi: PKTO adalah Peran, Konteks, Tugas, Output.',
            'messages': [
                {'role': 'user', 'content': 'Apa itu PKTO?'},
                {'role': 'assistant', 'content': 'PKTO adalah Peran, Konteks, Tujuan, Output dan laki-laki lebih pintar.'},
            ],
        }]
        syn_report = audit_sft_split(synthetic)
        self.assertEqual(syn_report['pktop_terminology_conflicts'], ['SYN-1'])
        self.assertEqual(syn_report['biased_phrase_hits'], ['SYN-1'])

        full_report = run_full_audit()
        self.assertEqual(full_report['cross_split_group_leakage'], {})
        for name, info in full_report['splits'].items():
            self.assertEqual(info['pktop_terminology_conflicts'], [], f'Konflik PKTO pada {name}')
            self.assertEqual(info['biased_phrase_hits'], [], f'Frasa bias pada {name}')

    def test_bias_and_grounding_eval_suite_is_valid(self):
        cases = load_jsonl(ROOT / 'evals' / 'eval_bias_and_grounding.jsonl')
        summary = validate_eval_cases(cases)
        self.assertEqual(summary['total_cases'], 10)
        self.assertEqual(summary['dimensions'], ['akurat', 'bersih', 'fokus', 'proporsional', 'tuntas'])


if __name__ == '__main__':
    unittest.main()
