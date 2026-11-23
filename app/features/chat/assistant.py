"""Qwen LLM assistant wrappers: local Hugging Face (auto bf16) and async vLLM."""

import asyncio
from contextlib import nullcontext
import json
import os
import threading

try:
    import httpx
except ImportError:
    httpx = None

try:
    import torch
except ImportError:
    torch = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = StoppingCriteriaList = TextIteratorStreamer = None

from app.core.config import load_record
from app.features.chat.sanitizer import StreamingSanitizer, sanitize_response
from app.features.chat.prompts import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_SAMPLING_CONFIG,
    SentenceBoundaryStoppingCriteria,
    build_generation_kwargs,
    fit_messages_to_budget,
    resolve_eos_token_ids,
    resolve_max_new_tokens,
    resolve_max_prompt_tokens,
)
from app.features.language.detector import (
    detect_language,
    format_user_message,
    is_wrong_language,
)
from app.features.language.locales import LOCALES, id as id_locale
from app.features.language.translator import (
    bilingual_queries,
    build_bilingual_query_messages,
    ensure_response_language,
    parse_bilingual_queries,
)


def _is_cuda_oom(exc):
    oom_cls = getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', ())
    return (oom_cls and isinstance(exc, oom_cls)) or 'out of memory' in str(exc).lower()


def _resolve_local_load_options(torch_mod):
    """Pick native bf16 (no 4-bit loss) on >=16 GB bf16 GPUs like L40S, else 4-bit."""
    use_bf16 = bool(torch_mod.cuda.is_bf16_supported())
    compute_dtype = torch_mod.bfloat16 if use_bf16 else torch_mod.float16
    mode = os.getenv('LLM_LOAD_IN_4BIT', 'auto').strip().lower()
    if mode in ('1', 'true', 'yes'):
        load_in_4bit = True
    elif mode in ('0', 'false', 'no'):
        load_in_4bit = False
    else:
        vram_bytes = torch_mod.cuda.get_device_properties(0).total_memory
        load_in_4bit = not (use_bf16 and vram_bytes >= 16 * (1024 ** 3))
    return compute_dtype, load_in_4bit


def _sampling_settings():
    return {
        'max_new_tokens': int(os.getenv('MAX_NEW_TOKENS', str(DEFAULT_MAX_NEW_TOKENS))),
        'do_sample': os.getenv(
            'GEN_DO_SAMPLE', str(DEFAULT_SAMPLING_CONFIG['do_sample'])
        ).strip().lower() not in ('0', 'false', 'no'),
        'temperature': float(
            os.getenv('GEN_TEMPERATURE', str(DEFAULT_SAMPLING_CONFIG['temperature']))
        ),
        'top_p': float(os.getenv('GEN_TOP_P', str(DEFAULT_SAMPLING_CONFIG['top_p']))),
        'min_p': float(os.getenv('GEN_MIN_P', str(DEFAULT_SAMPLING_CONFIG['min_p']))),
        'repetition_penalty': float(
            os.getenv('GEN_REPETITION_PENALTY', str(DEFAULT_SAMPLING_CONFIG['repetition_penalty']))
        ),
    }


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
        compute_dtype, load_in_4bit = _resolve_local_load_options(torch)
        model_kwargs = {
            'revision': record['revision'],
            'device_map': {'': 0},
            'dtype': compute_dtype,
            'attn_implementation': 'sdpa',
        }
        if load_in_4bit:
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(record['model_id'], **model_kwargs)
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
        cfg = _sampling_settings()
        self.max_new_tokens = cfg['max_new_tokens']
        self.do_sample = cfg['do_sample']
        self.temperature = cfg['temperature']
        self.top_p = cfg['top_p']
        self.min_p = cfg['min_p']
        self.repetition_penalty = cfg['repetition_penalty']
        configured_prompt = int(os.getenv('MAX_PROMPT_TOKENS', '0') or '0')
        self.max_prompt_tokens = resolve_max_prompt_tokens(
            model, max_new_tokens=self.max_new_tokens, configured=configured_prompt
        )

    def translate_queries(self, query, history=None):
        with self.gpu_lock:
            return bilingual_queries(query, self.model, self.tokenizer, history=history)

    def _sampling_for_request(self, persona=''):
        """Use Qwen3-Instruct-2507 recommended sampling when custom persona is active."""
        if persona and persona.strip() and self.do_sample:
            return {
                'temperature': max(self.temperature, 0.7),
                'top_p': 0.8,
                'top_k': 20,
                'min_p': 0.0,
            }
        return {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': None,
            'min_p': self.min_p,
        }

    def _adapter_context(self, persona=''):
        """Temporarily bypass neutral-tone LoRA so base Qwen3-Instruct can express student persona."""
        bypass_enabled = os.getenv('PERSONA_DISABLE_LORA', '1').strip().lower() not in (
            '0', 'false', 'no'
        )
        if persona and persona.strip() and bypass_enabled and hasattr(self.model, 'disable_adapter'):
            return self.model.disable_adapter()
        return nullcontext()

    def answer(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        recent = history[-2 * self.max_history_turns:]
        target_lang = lang or detect_language(question, recent)
        user_message = format_user_message(question, target_lang, persona=persona)
        sampling = self._sampling_for_request(persona)
        max_new_tokens = resolve_max_new_tokens(mode, self.max_new_tokens)
        eos_ids = resolve_eos_token_ids(self.tokenizer, self.model)
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
                student_name=student_name,
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
                    max_new_tokens=max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=sampling['temperature'],
                    top_p=sampling['top_p'],
                    top_k=sampling['top_k'],
                    min_p=sampling['min_p'],
                    repetition_penalty=self.repetition_penalty,
                    eos_token_id=eos_ids,
                )
                if StoppingCriteriaList is not None:
                    gen_kwargs['stopping_criteria'] = StoppingCriteriaList([
                        SentenceBoundaryStoppingCriteria(self.tokenizer, input_len, max_new_tokens)
                    ])
                with self.gpu_lock, self._adapter_context(persona), torch.inference_mode():
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
            checked = ensure_response_language(
                answer, target_lang, self.model, self.tokenizer,
                max_new_tokens=max_new_tokens,
            )
        return sanitize_response(checked)

    async def astream(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        """Stream generated tokens in real time via Hugging Face TextIteratorStreamer."""
        if TextIteratorStreamer is None:
            answer = await asyncio.to_thread(
                self.answer,
                question, mode, history, context, memory,
                lang, persona, student_name,
            )
            yield answer
            return

        recent = history[-2 * self.max_history_turns:]
        target_lang = lang or detect_language(question, recent)
        user_message = format_user_message(question, target_lang, persona=persona)
        sampling = self._sampling_for_request(persona)
        max_new_tokens = resolve_max_new_tokens(mode, self.max_new_tokens)
        eos_ids = resolve_eos_token_ids(self.tokenizer, self.model)
        messages, _ = fit_messages_to_budget(
            mode,
            context,
            user_message,
            history=recent,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            memory=memory,
            lang=target_lang,
            persona=persona,
            student_name=student_name,
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors='pt'
        )
        input_len = inputs['input_ids'].shape[1]
        inputs = {key: value.to('cuda:0') for key, value in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = build_generation_kwargs(
            self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=sampling['temperature'],
            top_p=sampling['top_p'],
            top_k=sampling['top_k'],
            min_p=sampling['min_p'],
            repetition_penalty=self.repetition_penalty,
            eos_token_id=eos_ids,
        )
        if StoppingCriteriaList is not None:
            gen_kwargs['stopping_criteria'] = StoppingCriteriaList([
                SentenceBoundaryStoppingCriteria(self.tokenizer, input_len, max_new_tokens)
            ])
        gen_kwargs.update(inputs)
        gen_kwargs['streamer'] = streamer
        del inputs

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _generate_worker(kwargs):
            try:
                with self.gpu_lock, self._adapter_context(persona), torch.inference_mode():
                    self.model.generate(**kwargs)
            except Exception as exc:
                try:
                    streamer.end()
                except Exception:
                    pass
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                kwargs.clear()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def _pump_worker():
            try:
                for chunk in streamer:
                    if chunk:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        gen_thread = threading.Thread(
            target=_generate_worker, args=(gen_kwargs,), daemon=True
        )
        pump_thread = threading.Thread(target=_pump_worker, daemon=True)
        gen_thread.start()
        pump_thread.start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item


class VLLMAssistant:
    """Async OpenAI-compatible vLLM client supporting continuous batching & streaming."""

    def __init__(self, base_url=None, model_name=None, api_key=None, tokenizer=None):
        if httpx is None:
            raise RuntimeError("Pustaka 'httpx' diperlukan untuk LLM_BACKEND=vllm.")
        record = load_record(os.getenv('MODEL_RECORD', 'model_record.json'))
        default_model = 'kka-lora' if os.getenv('MODEL_ADAPTER', '').strip() else record['model_id']
        self.base_url = (base_url or os.getenv('LLM_BASE_URL', 'http://127.0.0.1:8002/v1')).rstrip('/')
        self.model_name = (model_name or os.getenv('LLM_MODEL_NAME', default_model)).strip() or default_model
        self.api_key = (api_key or os.getenv('LLM_API_KEY', 'EMPTY')).strip() or 'EMPTY'
        self.tokenizer = tokenizer
        if self.tokenizer is None and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    record['model_id'], revision=record['revision'], local_files_only=True
                )
            except Exception:
                self.tokenizer = None
        self.max_history_turns = 4
        cfg = _sampling_settings()
        self.max_new_tokens = cfg['max_new_tokens']
        self.temperature = cfg['temperature'] if cfg['do_sample'] else 0.0
        self.top_p = cfg['top_p'] if cfg['do_sample'] else 1.0
        self.repetition_penalty = cfg['repetition_penalty']
        configured_prompt = int(os.getenv('MAX_PROMPT_TOKENS', '0') or '0')
        self.max_prompt_tokens = configured_prompt if configured_prompt > 0 else DEFAULT_MAX_PROMPT_TOKENS
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50, keepalive_expiry=30.0),
        )

    async def aclose(self):
        await self._client.aclose()

    def _headers(self):
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

    def _build_messages(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        recent = history[-2 * self.max_history_turns:]
        target_lang = lang or detect_language(question, recent)
        user_message = format_user_message(question, target_lang, persona=persona)
        messages, _ = fit_messages_to_budget(
            mode,
            context,
            user_message,
            history=recent,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            memory=memory,
            lang=target_lang,
            persona=persona,
            student_name=student_name,
        )
        return messages, target_lang

    async def _chat_complete(self, messages, *, max_tokens, temperature, top_p=1.0, repetition_penalty=1.0):
        payload = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'stream': False,
        }
        resp = await self._client.post(
            f'{self.base_url}/chat/completions', headers=self._headers(), json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get('choices') or []
        if not choices:
            return ''
        return (choices[0].get('message') or {}).get('content', '').strip()

    async def atranslate_queries(self, query, history=None):
        messages, condensed = build_bilingual_query_messages(query, history=history)
        try:
            raw = await self._chat_complete(messages, max_tokens=80, temperature=0.0)
        except Exception:
            return [condensed]
        return parse_bilingual_queries(raw, condensed)

    async def aanswer(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        messages, target_lang = self._build_messages(
            question, mode, history, context, memory=memory, lang=lang,
            persona=persona, student_name=student_name,
        )
        has_persona = bool(persona and persona.strip())
        temp = max(self.temperature, 0.7) if (has_persona and self.temperature > 0) else self.temperature
        top_p = 0.8 if (has_persona and self.temperature > 0) else self.top_p
        max_new_tokens = resolve_max_new_tokens(mode, self.max_new_tokens)
        answer = await self._chat_complete(
            messages,
            max_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=self.repetition_penalty,
        )
        if answer and is_wrong_language(answer, target_lang):
            target_name = LOCALES.get(target_lang, id_locale).LANGUAGE_NAME
            rewrite_msgs = [
                {
                    'role': 'system',
                    'content': (
                        f'Translate the assistant response into natural {target_name}. '
                        'Preserve numbered or bulleted steps, reasoning, and any '
                        'closing reflection questions without adding citation tags like [S1]. '
                        'Return only the translated response.'
                    ),
                },
                {'role': 'user', 'content': answer},
            ]
            rewritten = await self._chat_complete(
                rewrite_msgs, max_tokens=self.max_new_tokens, temperature=0.0
            )
            if rewritten and not is_wrong_language(rewritten, target_lang):
                return sanitize_response(rewritten)
        return sanitize_response(answer)

    async def astream(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        messages, _ = self._build_messages(
            question, mode, history, context, memory=memory, lang=lang,
            persona=persona, student_name=student_name,
        )
        has_persona = bool(persona and persona.strip())
        temp = max(self.temperature, 0.7) if (has_persona and self.temperature > 0) else self.temperature
        top_p = 0.8 if (has_persona and self.temperature > 0) else self.top_p
        max_new_tokens = resolve_max_new_tokens(mode, self.max_new_tokens)
        payload = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_new_tokens,
            'temperature': temp,
            'top_p': top_p,
            'repetition_penalty': self.repetition_penalty,
            'stream': True,
        }
        sanitizer = StreamingSanitizer()
        async with self._client.stream(
            'POST', f'{self.base_url}/chat/completions', headers=self._headers(), json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith('data:'):
                    continue
                data_str = line[5:].strip()
                if data_str == '[DONE]':
                    break
                chunk = json.loads(data_str)
                choices = chunk.get('choices') or []
                if not choices:
                    continue
                delta = (choices[0].get('delta') or {}).get('content')
                if delta:
                    cleaned = sanitizer.feed(delta)
                    if cleaned:
                        yield cleaned
        tail = sanitizer.flush()
        if tail:
            yield tail


def create_assistant():
    backend = os.getenv('LLM_BACKEND', 'local').strip().lower()
    if backend == 'vllm':
        return VLLMAssistant()
    return LocalAssistant()
