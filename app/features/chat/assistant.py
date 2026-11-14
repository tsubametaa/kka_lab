"""Local Qwen LLM assistant inference wrapper."""

import os
import threading

try:
    import torch
except ImportError:
    torch = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None

from app.core.config import load_record
from app.features.chat.prompts import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_SAMPLING_CONFIG,
    build_generation_kwargs,
    fit_messages_to_budget,
    make_system,
    resolve_max_prompt_tokens,
)
from app.features.language.detector import detect_language, format_user_message
from app.features.language.translator import ensure_response_language


def _is_cuda_oom(exc):
    oom_cls = getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', ())
    return (oom_cls and isinstance(exc, oom_cls)) or 'out of memory' in str(exc).lower()


class LocalAssistant:
    def __init__(self):
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError('CUDA belum terdeteksi. Periksa driver dan PyTorch GPU.')
        if AutoModelForCausalLM is None:
            raise RuntimeError("Pustaka 'transformers' belum terpasang.")
        record = load_record(os.getenv('MODEL_RECORD', 'model_record.json'))
        self.tokenizer = AutoTokenizer.from_pretrained(
            record['model_id'], revision=record['revision']
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            record['model_id'], revision=record['revision'],
            quantization_config=quant, device_map={'': 0},
            dtype=torch.float16, attn_implementation='sdpa',
        )
        adapter = os.getenv('MODEL_ADAPTER', '').strip()
        if adapter:
            adapter_record = load_record(os.path.join(adapter, 'model_record.json'))
            if adapter_record != record:
                raise ValueError('Model/revisi adaptor berbeda dari model lokal.')
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval()
        self.model = model
        self.gpu_lock = threading.RLock()
        self.max_history_turns = 4
        self.max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', str(DEFAULT_MAX_NEW_TOKENS)))
        self.do_sample = os.getenv(
            'GEN_DO_SAMPLE', str(DEFAULT_SAMPLING_CONFIG['do_sample'])
        ).strip().lower() not in ('0', 'false', 'no')
        self.temperature = float(
            os.getenv('GEN_TEMPERATURE', str(DEFAULT_SAMPLING_CONFIG['temperature']))
        )
        self.top_p = float(
            os.getenv('GEN_TOP_P', str(DEFAULT_SAMPLING_CONFIG['top_p']))
        )
        self.min_p = float(
            os.getenv('GEN_MIN_P', str(DEFAULT_SAMPLING_CONFIG['min_p']))
        )
        self.repetition_penalty = float(
            os.getenv('GEN_REPETITION_PENALTY', str(DEFAULT_SAMPLING_CONFIG['repetition_penalty']))
        )
        configured_prompt = int(os.getenv('MAX_PROMPT_TOKENS', '0') or '0')
        self.max_prompt_tokens = resolve_max_prompt_tokens(
            model, max_new_tokens=self.max_new_tokens, configured=configured_prompt
        )

    def answer(self, question, mode, history, context, memory='', lang=None, persona=''):
        recent = history[-2 * self.max_history_turns:]
        target_lang = lang or detect_language(question, recent)
        user_message = format_user_message(question, target_lang)
        budget = self.max_prompt_tokens
        while True:
            messages, prompt_tokens = fit_messages_to_budget(
                mode,
                context,
                user_message,
                history=recent,
                tokenizer=self.tokenizer,
                max_prompt_tokens=budget,
                memory=memory,
                lang=target_lang,
                persona=persona,
            )
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors='pt'
            )
            input_len = inputs['input_ids'].shape[1]
            inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
            try:
                gen_kwargs = build_generation_kwargs(
                    self.tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    min_p=self.min_p,
                    repetition_penalty=self.repetition_penalty,
                )
                with self.gpu_lock, torch.inference_mode():
                    output = self.model.generate(**inputs, **gen_kwargs)
                break
            except RuntimeError as exc:
                del inputs
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not _is_cuda_oom(exc) or prompt_tokens <= 1536:
                    raise
                budget = max(1536, min(budget, prompt_tokens) // 2)
        new_tokens = output[0, input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        with self.gpu_lock:
            return ensure_response_language(
                answer, target_lang, self.model, self.tokenizer,
                max_new_tokens=self.max_new_tokens,
            )
