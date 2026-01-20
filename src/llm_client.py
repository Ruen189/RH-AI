from typing import List, Optional, Union
import gc
import os

import torch
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


class LlamaClient:
    def __init__(
        self,
        adapter_dir: Optional[str] = None,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.85,
        max_lora_rank: int = 32,
        enforce_eager: bool = False,
    ):
        # Можно не snapshot_download и просто передать MODEL_NAME в vLLM.
        # Но оставим как в вашем стиле, чтобы путь был локальный.
        print("[LLM] Проверяем и докачиваем модель (snapshot_download)...")
        cache_dir = snapshot_download(repo_id=MODEL_NAME)
        print(f"[LLM] Загружаем модель из {cache_dir}")

        self.lora_request: Optional[LoRARequest] = None
        enable_lora = False

        if adapter_dir:
            adapter_dir = os.path.abspath(os.path.expanduser(adapter_dir))
            enable_lora = True
            self.lora_request = LoRARequest("lora_adapter", 1, adapter_dir)
            print(f"[LLM] LoRA включена: {adapter_dir}")

        # AWQ INT4
        # Для AWQ-моделей в vLLM нужно указать quantization="awq". [web:216]
        self.llm = LLM(
            model=cache_dir,
            quantization="awq",
            dtype="half",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            enforce_eager=enforce_eager,
        )

        print("[LLM] LlamaClient (vLLM, AWQ INT4) готов к работе.")

    def generate(
        self,
        prompts: Union[str, List[str]],
        *,
        max_new_tokens: int = 128,
        min_new_tokens: int = 2,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_tqdm: bool = False,
    ) -> Union[str, List[str]]:
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else prompts

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            min_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.llm.generate(
            prompt_list,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=self.lora_request,
        )

        texts = [out.outputs[0].text.strip() for out in outputs]
        return texts[0] if is_single else texts

    # совместимость со старым кодом
    def ask_one(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.9,
            use_tqdm=False,
        )

    # совместимость со старым кодом (batch_size игнорируется — vLLM батчит сам)
    def ask_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ) -> List[str]:
        _ = batch_size
        return self.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=1.0,
            use_tqdm=True,
        )

    def close(self):
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


llama_client = None
llama_adapter_dir = None


def get_llama(adapter_dir: Optional[str] = None):
    global llama_client, llama_adapter_dir
    if llama_client is None or adapter_dir != llama_adapter_dir:
        llama_adapter_dir = adapter_dir
        llama_client = LlamaClient(adapter_dir=adapter_dir)
    return llama_client


def reset_llama():
    global llama_client, llama_adapter_dir
    try:
        if llama_client is not None:
            llama_client.close()
    except Exception:
        pass

    llama_client = None
    llama_adapter_dir = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
