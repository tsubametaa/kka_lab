"""Tests for training/train_lora.py CLI defaults, presets, and flag validation."""

import unittest

from training.train_lora import apply_preset, build_parser, parse_target_modules


class TrainLoraCliTests(unittest.TestCase):
    def test_default_optimization_args(self):
        args = build_parser().parse_args(['--data', 'd.jsonl', '--out', 'o'])
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.eval_strategy, 'steps')
        self.assertEqual(args.eval_steps, 25)
        self.assertEqual(args.early_stopping_patience, 3)

    def test_rtx3050_preset(self):
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

    def test_regularization_flags_and_validation(self):
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
