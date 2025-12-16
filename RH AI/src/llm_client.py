## src/llm_client.py
from typing import List, Optional
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import os
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


class LlamaClient:
    def __init__(self, device: str = "cuda:0", adapter_dir: Optional[str] = None):
        print("[LLM] Проверяем и докачиваем модель (snapshot_download)...")
        cache_dir = snapshot_download(
            repo_id=MODEL_NAME,
        )
        print(f"[LLM] Загружаем модель из {cache_dir}")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cache_dir,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

        if adapter_dir:
            adapter_dir = os.path.expanduser(adapter_dir)
            adapter_dir = os.path.abspath(adapter_dir)
            from peft import PeftModel
            print(f"[LLM] Подключаем LoRA адаптер: {adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
        )

        print("[LLM] LlamaClient готов к работе.")

    # один запрос
    def ask_one(self, prompt: str, max_new_tokens: int = 256) -> str:
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=True,
            top_p=0.9,
            temperature=0.4,  # низкая температура → стабильный ответ
            return_full_text=False,
        )
        text = outputs[0]["generated_text"]
        return text.strip()

    # батч запросов
    def ask_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ) -> List[str]:
        outputs = self.pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=False,
            batch_size=batch_size,
        )
        results = []
        for result in outputs:
            # для батча pipeline возвращает List[Dict], а не List[List[Dict]] при return_full_text=False
            if isinstance(result, list):
                result = result[0]
            text = result["generated_text"]
            results.append(text.strip())
        return results
    def close(self):
        # удаляем тяжелые объекты
        if hasattr(self, "pipe"):
            del self.pipe
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


llama_client = None
llama_adapter_dir = None

def get_llama(adapter_dir: Optional[str] = None):
    global llama_client, llama_adapter_dir

    # если клиента нет или адаптер поменялся — пересоздаём
    if llama_client is None or adapter_dir != llama_adapter_dir:
        llama_adapter_dir = adapter_dir
        llama_client = LlamaClient(adapter_dir=adapter_dir)
    return llama_client

def reset_llama():
    global llama_client, llama_adapter_dir
    llama_client = None
    llama_adapter_dir = None

