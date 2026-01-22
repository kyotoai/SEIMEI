
"""LoRA reward-model training helpers with GRPO and W&B logging."""
from __future__ import annotations
import os
import random
import math
import re
import requests
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import RewardConfig, RewardTrainer
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from accelerate.utils import broadcast_object_list
import torch.distributed as dist
from .utils import extract_int, extract_text


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PARALLEL"] = "1"
# dist.init_process_group("gloo")  #use gloo backend if NCCL is misbehaving (<2 GPU)


random.seed(42)

# --- DDP: pin each rank to its CUDA device early ---
local_rank_str = os.environ.get("LOCAL_RANK")
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(local_rank_str) if local_rank_str is not None else None
is_multi_gpu = world_size > 1
if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
# ---------------------------------------------------


print(f"--> Starting process with rank: {os.environ.get('RANK')}, "
      f"local_rank: {os.environ.get('LOCAL_RANK')}, "
      f"visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("torch.cuda.is_available()", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

__all__ = ["make_dataset_list", "train_reward_model"]

_PROMPT_TEMPLATE = (
    "Give me relevant score between query and sentence;\n\n"
    "Query:{query}\n\n"
    'Sentence:```{sentence}```'
)

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available
from trl.trainer.utils import compute_accuracy


def is_main_process() -> bool:
    # If torch.distributed is initialized, use its rank
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Fallback to env var set by launchers (Accelerate/torchrun)
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass
    # Single-process case
    return True


def format_query_for_rmsearch(body: str) -> str:
    text = (body or "").strip()
    if text.startswith("<query>") and "</query>" in text:
        return text
    if not text:
        text = "[missing query context]"
    return f"<query>\n{text}\n</query>"


def format_key_for_rmsearch(content: str, *, tags: Optional[Sequence[Any]] = None) -> str:
    tag_values: List[str] = []
    for tag in tags or []:
        tag_text = str(tag).strip()
        if tag_text:
            tag_values.append(tag_text)

    base_text = (content or "").strip()
    if not base_text:
        base_text = "[missing key text]"

    if base_text.startswith("<key>") and "</key>" in base_text:
        formatted = base_text
    else:
        lines = [base_text]
        if tag_values:
            lines.append(f"Tags: {', '.join(tag_values)}")
        joined = "\n".join(lines)
        formatted = f"<key>\n{joined}\n</key>"

    formatted = formatted.strip()
    if "Query-Key Relevance Score:" not in formatted:
        formatted = f"{formatted}\n\n\nQuery-Key Relevance Score:"
    return formatted


def _extract_tag_block(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
        return ""
    start = text.find(start_tag) + len(start_tag)
    end = text.find(end_tag, start)
    if end < 0:
        return ""
    return text[start:end].strip()


def _split_key_and_tags(key_text: str) -> Tuple[str, List[str]]:
    if not key_text:
        return "", []
    lines = [line.rstrip() for line in key_text.splitlines()]
    if not lines:
        return "", []
    last = lines[-1].strip()
    if last.lower().startswith("tags:"):
        tag_blob = last[len("tags:"):].strip()
        tags = [item.strip() for item in tag_blob.split(",") if item.strip()]
        return "\n".join(lines[:-1]).strip(), tags
    return key_text.strip(), []


def _extract_rm_query_key(prompt: str) -> Tuple[str, Dict[str, Any]]:
    query_text = _extract_tag_block(prompt, "query")
    key_block = _extract_tag_block(prompt, "key")
    key_text, tags = _split_key_and_tags(key_block)
    if not query_text:
        query_text = prompt.strip()
    if not key_text:
        key_text = key_block.strip() if key_block else ""
    return query_text, {"key": key_text, "tags": tags}


def _coerce_reward_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    score = max(0.0, min(10.0, score))
    return float(int(round(score)))


class RMSearchClient:
    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self.base_url = str(base_url or "").strip()
        self.timeout = float(timeout)

    def score(self, query: str, keys: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        if not self.base_url or not keys:
            return []
        formatted_query = format_query_for_rmsearch(query)
        payload_keys: List[str] = []
        text_map: Dict[str, int] = {}
        for idx, item in enumerate(keys):
            if not isinstance(item, dict):
                payload_keys.append(format_key_for_rmsearch(""))
                continue
            key_text = str(item.get("key") or "").strip()
            tags = item.get("tags") or []
            formatted = format_key_for_rmsearch(key_text, tags=tags)
            payload_keys.append(formatted)
            text_map[formatted] = idx

        payload: Dict[str, Any] = {
            "queries": [formatted_query],
            "keys": payload_keys,
            "k": len(payload_keys),
        }

        api_key = os.getenv("KYOTOAI_API_KEY")
        if not api_key:
            raise RuntimeError("KYOTOAI_API_KEY environment variable is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from RMSearch: {exc}") from exc

        scores: List[Optional[float]] = [None] * len(payload_keys)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("results") or data.get("queries") or []
            if isinstance(entries, dict):
                entries = [entries]
        else:
            entries = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            keys_block = entry.get("keys") or []
            if not isinstance(keys_block, Sequence):
                continue
            for item in keys_block:
                if not isinstance(item, dict):
                    continue
                key_idx = item.get("key_id")
                key_text = item.get("key")
                target_idx = None
                if isinstance(key_idx, int) and 0 <= key_idx < len(scores):
                    target_idx = key_idx
                elif isinstance(key_text, str) and key_text in text_map:
                    target_idx = text_map[key_text]
                if target_idx is None:
                    continue
                scores[target_idx] = _coerce_reward_score(item.get("relevance"))
        return scores


class CustomGRPOTrainer(Trainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(
        self,
        *args,
        grpo_clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        rm_client: Optional[RMSearchClient] = None,
        **kwargs,
    ):
        super().__init__(compute_metrics=None, *args, **kwargs)
        self.grpo_clip_eps = float(grpo_clip_eps)
        self.kl_coef = float(kl_coef)
        self.rm_client = rm_client
        self._rm_cache: Dict[Tuple[str, Tuple[Tuple[str, Tuple[str, ...]], ...]], torch.Tensor] = {}


    # --- ensure DistributedSampler on multi-GPU ---
    def get_train_dataloader(self):
        dataset = self.train_dataset
        if dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        if self.args.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            sampler = RandomSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return None

        if self.args.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    # def _zero_loss_like_model(model):
    # # attach to the graph cheaply
    #     return next(model.parameters()).sum() * 0.0

    def compute_loss(  # GPT5
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """
        Expected shapes from collator:
        input_ids:      [B, K, L]
        attention_mask: [B, K, L]
        scores:         [B, K] (float, NaN for missing)
        """
        inputs = self._prepare_inputs(inputs)

        input_ids = inputs["input_ids"]  # [B, K, L]
        attention_mask = inputs["attention_mask"]  # [B, K, L]
        scores = inputs["scores"]  # [B, K]
        rm_query = inputs.get("rm_query")
        rm_keys = inputs.get("rm_keys")

        B, K, L = input_ids.shape

        input_ids_flat = input_ids.reshape(B * K, L)
        attention_mask_flat = attention_mask.reshape(B * K, L)

        logits_flat = model(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            return_dict=True,
        )["logits"].squeeze(-1)
        logits = logits_flat.view(B, K).float()

        log_probs = torch.log_softmax(logits, dim=1)
        old_log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)

        rewards = scores.to(logits.device, dtype=logits.dtype)
        valid_mask = torch.isfinite(rewards)
        valid_counts = valid_mask.sum(dim=1, keepdim=True)
        safe_counts = valid_counts.clamp(min=1)
        rewards_zeroed = rewards.masked_fill(~valid_mask, 0.0)
        mean = rewards_zeroed.sum(dim=1, keepdim=True) / safe_counts
        variance = ((rewards - mean).masked_fill(~valid_mask, 0.0) ** 2).sum(dim=1, keepdim=True) / safe_counts
        std = torch.sqrt(variance + 1e-8)

        advantages = torch.where(valid_mask, (rewards - mean) / std, torch.zeros_like(rewards))
        stable_mask = (valid_counts > 1) & (std > 1e-6)
        advantages = torch.where(stable_mask, advantages, torch.zeros_like(advantages))

        clipped = torch.clamp(ratio, 1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps)
        pg_terms = torch.minimum(ratio * advantages, clipped * advantages)
        if valid_mask.any():
            policy_loss = -pg_terms.masked_fill(~valid_mask, 0.0).sum() / valid_mask.sum().clamp(min=1)
        else:
            policy_loss = logits.sum() * 0.0

        kl_loss = logits.sum() * 0.0
        if self.kl_coef > 0 and self.rm_client is not None and rm_query is not None and rm_keys is not None:
            ref_log_probs, kl_mask = self._get_ref_log_probs_batch(
                rm_query=rm_query,
                rm_keys=rm_keys,
                target_len=K,
                device=logits.device,
                dtype=logits.dtype,
            )
            if ref_log_probs is not None and kl_mask is not None and kl_mask.sum() > 0:
                probs = torch.softmax(logits, dim=1)
                kl_values = (probs * (log_probs - ref_log_probs)).sum(dim=1)
                kl_loss = (kl_values * kl_mask).sum() / kl_mask.sum().clamp(min=1)

        total_loss = policy_loss + self.kl_coef * kl_loss

        if return_outputs:
            return total_loss, {
                "logits": logits,
                "advantages": advantages,
            }
        return total_loss

    def _rm_cache_key(
        self,
        query: str,
        keys: Sequence[Dict[str, Any]],
    ) -> Tuple[str, Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        key_items: List[Tuple[str, Tuple[str, ...]]] = []
        for item in keys or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("key") or "")
            tags = tuple(str(tag) for tag in (item.get("tags") or []))
            key_items.append((text, tags))
        return str(query or ""), tuple(key_items)

    def _get_ref_log_probs(
        self,
        *,
        query: str,
        keys: Sequence[Dict[str, Any]],
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        cache_key = self._rm_cache_key(query, keys)
        cached = self._rm_cache.get(cache_key)
        if cached is not None:
            return cached.to(device=device, dtype=dtype)

        if self.rm_client is None:
            return None
        try:
            scores = self.rm_client.score(query, keys)
        except Exception:
            return None
        if not scores:
            return None

        padded = []
        for value in list(scores)[:target_len]:
            padded.append(_coerce_reward_score(value))
        if len(padded) < target_len:
            padded.extend([None] * (target_len - len(padded)))
        score_values = [
            value if value is not None else 0.0
            for value in padded
        ]
        score_tensor = torch.tensor(score_values, dtype=torch.float32)
        ref_log_probs = torch.log_softmax(score_tensor, dim=0)
        self._rm_cache[cache_key] = ref_log_probs.cpu()
        return ref_log_probs.to(device=device, dtype=dtype)

    def _get_ref_log_probs_batch(
        self,
        *,
        rm_query: Sequence[Any],
        rm_keys: Sequence[Any],
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.rm_client is None:
            return None, None
        ref_list: List[torch.Tensor] = []
        mask_list: List[float] = []
        for query, keys in zip(rm_query, rm_keys):
            if not isinstance(keys, Sequence):
                keys = []
            ref = self._get_ref_log_probs(
                query=str(query or ""),
                keys=[dict(item) for item in keys if isinstance(item, dict)],
                target_len=target_len,
                device=device,
                dtype=dtype,
            )
            if ref is None:
                mask_list.append(0.0)
                ref_list.append(torch.zeros(target_len, device=device, dtype=dtype))
            else:
                mask_list.append(1.0)
                if ref.numel() != target_len:
                    padded = torch.zeros(target_len, device=device, dtype=dtype)
                    length = min(target_len, ref.numel())
                    padded[:length] = ref[:length]
                    ref = padded
                ref_list.append(ref.to(device=device, dtype=dtype))
        if not ref_list:
            return None, None
        mask = torch.tensor(mask_list, device=device, dtype=dtype)
        return torch.stack(ref_list, dim=0), mask

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return loss, None, None

        logits = outputs.get("logits")
        if logits is not None:
            logits = logits.detach()
        return loss.detach(), logits, None

    def train(self, *args, **kwargs): # You need this because it will use RewardTrainer compute_loss method without this. To use a subclass function, some method in the subclass must be called from main directly. 
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)
        #return super(RewardTrainer, self).evaluate(*args, **kwargs)
        #return super().evaluate(num_print_samples=1, *args, **kwargs) # this fell in an error for some reason



'''
# CustomRewardTrainer example
class CustomRewardTrainer(RewardTrainer):
    _tag_names = ["trl", "reward-trainer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs): # You need this because it will use RewardTrainer compute_loss method without this. To use a subclass function, some method in the subclass must be called from main directly. 
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return super().evaluate(num_print_samples=1, *args, **kwargs)

'''

def _truncate_message(content: str, limit: int) -> str:
    text = str(content)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_grpo_group(
    example: Dict[str, object],
    tokenizer,
    *,
    max_length: int,
    max_characters: int,
) -> Dict[str, List[int]]:
    batch = example.get("batch")
    scores = example.get("scores") or []

    if not isinstance(batch, list):
        raise ValueError("'batch' must be a list.")

    prompts: List[str] = []
    rm_query = ""
    rm_keys: List[Dict[str, Any]] = []
    for batch_dict in batch:
        if not isinstance(batch_dict, dict):
            continue
        prompt = tokenizer.apply_chat_template(batch_dict["msg"], tokenize=False)
        prompts.append(prompt)
        query_text, key_payload = _extract_rm_query_key(prompt)
        if not rm_query and query_text:
            rm_query = query_text
        rm_keys.append(key_payload)

    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "scores": scores,
        "rm_query": rm_query,
        "rm_keys": rm_keys,
    }


def _build_dataset_split(
    records: Sequence[Dict[str, object]],
    tokenizer,
    *,
    max_length: int,
    max_characters: int,
   sample_ratio: float = 1.0,
) -> Optional[Dataset]:
    if not records:
        return None


    dataset = Dataset.from_list(list(records))

    # 2) Deterministic global subsample (once) so every rank picks the SAME indices
    if sample_ratio < 1.0:
        seed = int(os.environ.get("GLOBAL_SEED", "42"))
        rng = random.Random(seed)

        target = max(1, int(math.floor(len(dataset) * sample_ratio)))

        # If using multi-GPU, make subset size divisible by world size to avoid length mismatch
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            # ensure at least 1 per rank
            target = max(ws, (target // ws) * ws)

        indices = rng.sample(range(len(dataset)), target)
        dataset = dataset.select(indices)



    print(f"[Rank {dist.get_rank() if dist.is_available() and dist.is_initialized() else 0}] "
          f"After sampling+shard len={len(dataset)}")
    print("After sampling",len(dataset))

    def format_example(example: Dict[str, object]) -> Dict[str, List[int]]:
        return _format_grpo_group(
            example,
            tokenizer,
            max_length=max_length,
            max_characters=max_characters,
        )

    tokenized = dataset.map(format_example)

    keep_columns = {
        "input_ids",
        "attention_mask",
        "scores",
        "rm_query",
        "rm_keys",
    }

    columns_to_remove = [column for column in tokenized.column_names if column not in keep_columns]
    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)

    return tokenized


def make_dataset_list(
    results: Sequence[Dict[str, object]],
    *,
    sentences: Sequence[str],
) -> List[Dict[str, object]]:
    """Convert judge outputs into chat-format preference pairs."""

    dataset_list: List[Dict[str, object]] = []

    for result in results:
        output = str(result.get("output", ""))
        sentence_ids = list(result.get("sentence_ids", []))
        question = str(result.get("question", ""))

        chosen_id = extract_text(output, "ID")
        if chosen_id is None:
            chosen_id = extract_int(output[-10:])
        try:
            chosen_idx = int(chosen_id)
        except Exception:
            continue
        if chosen_idx not in (1, 2):
            continue

        if len(sentence_ids) < 2:
            continue

        chosen_sentence_id = sentence_ids[0] if chosen_idx == 1 else sentence_ids[1]
        rejected_sentence_id = sentence_ids[1] if chosen_idx == 1 else sentence_ids[0]
        if chosen_sentence_id >= len(sentences) or rejected_sentence_id >= len(sentences):
            continue

        dataset_list.append(
            {
                "chosen_msg": [
                    {
                        "role": "user",
                        "content": _PROMPT_TEMPLATE.format(
                            query=question,
                            sentence=sentences[chosen_sentence_id],
                        ),
                    }
                ],
                "rejected_msg": [
                    {
                        "role": "user",
                        "content": _PROMPT_TEMPLATE.format(
                            query=question,
                            sentence=sentences[rejected_sentence_id],
                        ),
                    }
                ],
                "chosen_sentence_id": chosen_sentence_id,
                "rejected_sentence_id": rejected_sentence_id,
            }
        )

    # dataset_list (list): preference pairs used by TRL, where each element is
    #   {
    #     "chosen_msg": [{"role": "user", "content": "<prompt with positive sentence>"}],
    #     "rejected_msg": [{"role": "user", "content": "<prompt with negative sentence>"}],
    #     "chosen_sentence_id": <index of the preferred sentence>,
    #     "rejected_sentence_id": <index of the less relevant sentence>
    #   }
    return dataset_list


from transformers import TrainerCallback

class PreciseLrCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # kwargs usually has optimizer and lr_scheduler
        lr_scheduler = kwargs.get("lr_scheduler", None)
        optimizer = kwargs.get("optimizer", None)

        if lr_scheduler is not None:
            lr = lr_scheduler.get_last_lr()[0]
        elif optimizer is not None:
            lr = optimizer.param_groups[0]["lr"]
        else:
            return

        # print to console with high precision
        print(f"[step {state.global_step}] lr={lr:.10f}")


def train_reward_model(
    dataset_list_train: Sequence[Dict[str, object]],
    *,
    dataset_list_test: Sequence[Dict[str, object]] | None = None,
    model_name: str,
    output_dir: Path = Path("./rm_model"),
    max_length: int = 1000,
    max_characters: int = 4000,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 2,
    evaluation_steps: int = 80,
    save_steps: int = 40,
    logging_steps: int = 1,
    num_train_epochs: int = 5,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    load_in_8bit: bool = False,
    grpo_clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    rm_url: Optional[str] = None,
    rm_timeout: float = 10.0,
) -> None:
    """Train a reward model using GRPO with LoRA adapters."""

    output_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs: Dict[str, object] = {"num_labels": 1}


    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_bos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    if load_in_8bit:
        if not torch.cuda.is_available():
            raise RuntimeError("8-bit quantization requires CUDA-enabled hardware.")

        local_rank = os.environ.get("LOCAL_RANK")
        print("WORLD size", world_size)
        # device_map = {"": int(local_rank)} if local_rank is not None else "auto"
        device_map = {"": int(local_rank)} if (world_size > 1 and local_rank is not None) else "cuda:0"
    
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs.update({
            "quantization_config": bnb_config,
            "device_map": device_map,
            "torch_dtype": torch.float16  
        })
    else:
         model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",
        **model_kwargs
    )

    if load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "k_proj",
            "q_proj",
            "o_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
        ],
        layers_to_transform=[25, 26, 27],
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False  

    if not load_in_8bit:
        model.enable_input_require_grads()

    train_dataset = _build_dataset_split(
        dataset_list_train,
        tokenizer,
        max_length=max_length,
        max_characters=max_characters,
    )
    if train_dataset is None:
        raise ValueError("Training dataset is empty; provide at least one preference pair.")

    eval_dataset = _build_dataset_split(
        dataset_list_test or [],
        tokenizer,
        max_length=max_length,
        max_characters=max_characters,
    )

    rm_client = None
    if rm_url:
        rm_client = RMSearchClient(rm_url, timeout=rm_timeout)

    report_to: List[str] = []
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name
        if wandb_tags:
            os.environ["WANDB_TAGS"] = ",".join(wandb_tags)
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        if not is_main_process():
            os.environ["WANDB_SILENT"] = "true"
        report_to = ["wandb"]


    evaluation_strategy = "steps" if eval_dataset is not None else "no"


  

    print("taining loop start")


    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=wandb_run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy=evaluation_strategy,
        eval_steps=evaluation_steps,
        optim="paged_adamw_8bit",   # if bitsandbytes is present
        torch_compile=True, 
        learning_rate=8e-5,              # "high" for LoRA;
        weight_decay=0.05,               # small regularization helps with high LR
        lr_scheduler_type="cosine",      # good default for high LR
        # lr_scheduler_kwargs={"num_cycles": 4},
        warmup_steps=40,           
        max_grad_norm=1.0, 
        fp16=True,
        eval_on_start=bool(eval_dataset),
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=max(2, os.cpu_count() // 2),
        # dataloader_num_workers=0,
        dataloader_pin_memory=True,
        disable_tqdm=not is_main_process(),

    )

    def custom_data_collator(features):  #GPT5
        """
        Produces:
        input_ids:      [B, K, L] (long)
        attention_mask: [B, K, L] (long)
        scores:         [B, K] (float, NaN for missing)
        rm_query:       list[str]
        rm_keys:        list[list[dict]]
        """
        batch = {}

        # Tensor fields
        tensor_fields = ["input_ids", "attention_mask"]
        for field in tensor_fields:
            batch[field] = torch.stack(
                [torch.as_tensor(f[field], dtype=torch.long) for f in features],  # -> [B, K, L]
                dim=0,
            )

        B, K, _ = batch["input_ids"].shape
        scores_tensor = torch.full((B, K), float("nan"), dtype=torch.float)
        for idx, feature in enumerate(features):
            raw_scores = feature.get("scores") or []
            if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes, bytearray)):
                raw_scores = []
            for jdx, value in enumerate(list(raw_scores)[:K]):
                score = _coerce_reward_score(value)
                if score is None:
                    continue
                scores_tensor[idx, jdx] = score
        batch["scores"] = scores_tensor

        # Non-tensor fields (kept as Python objects per sample)
        batch["rm_query"] = [f.get("rm_query", "") for f in features]
        batch["rm_keys"] = [f.get("rm_keys", []) for f in features]

        return batch


    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
        grpo_clip_eps=grpo_clip_eps,
        kl_coef=kl_coef,
        rm_client=rm_client,
    )



    dl = trainer.get_train_dataloader()
    s = getattr(dl, "sampler", None)
    print(f"[Rank {os.environ.get('LOCAL_RANK','?')}] sampler={type(s).__name__} "
        f"num_replicas={getattr(s, 'num_replicas', None)} rank={getattr(s, 'rank', None)}")

    print(f"[Rank {os.environ.get('LOCAL_RANK','?')}] steps_per_epoch={len(dl)}")
# -------------------------------------------

    # ---- Log clipped grad norm to W&B (works with TrainingArguments max_grad_norm) ----
    def install_wandb_gradnorm_logger(trainer, key_prefix="grad_norm"):
        try:
            import wandb
        except Exception:
            wandb = None

        accel = trainer.accelerator                     # used internally by Trainer
        orig_clip = accel.clip_grad_norm_               # original function

        def wrapped_clip_grad_norm_(params, max_norm, *args, **kwargs):
            # Ensure we only touch real grads from trainable params (LoRA etc.)
            if not isinstance(params, (list, tuple)):
                params = list(params)
            params = [p for p in params if (p is not None and p.requires_grad and p.grad is not None)]

            # Pre-clip norm (the value Accelerate returns)
            pre = orig_clip(params, max_norm, *args, **kwargs)

            # Post-clip norm: recompute after clipping
            post = float("nan")
            try:
                with torch.no_grad():
                    norms = [p.grad.detach().data.norm(2) for p in params]
                    if norms:
                        post = torch.norm(torch.stack(norms), 2).item()
            except Exception:
                pass

            # Log only on main process, and only if W&B is enabled
            if trainer.is_world_process_zero() and wandb is not None and ("wandb" in (trainer.args.report_to or [])):
                payload = {
                    f"{key_prefix}_pre_clip": float(pre),    # can be inf early on
                    f"{key_prefix}_post_clip": float(post),  # should be <= max_norm
                    "max_grad_norm": float(max_norm),
                }
                # If fp16 is used, this helps debug scaler behavior
                scaler = getattr(trainer.accelerator, "scaler", None)
                if scaler is not None:
                    try:
                        payload["grad_scale"] = float(scaler.get_scale())
                    except Exception:
                        pass
                wandb.log(payload)

            return pre

        accel.clip_grad_norm_ = wrapped_clip_grad_norm_

    trainer.add_callback(PreciseLrCallback())
    install_wandb_gradnorm_logger(trainer) 
    # trainer.train(resume_from_checkpoint="/workspace/exp2/model1/checkpoint-280") #for chekpoint [not tested]
    trainer.train()


    if eval_dataset is not None:
        trainer.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model using LoRA adapters.")
    parser.add_argument(
        "--dataset-list-train",
        type=Path,
        required=True,
        help="Path to dataset_list_train.json produced by judge_dataset.py.",
    )
    parser.add_argument(
        "--dataset-list-test",
        type=Path,
        help="Optional dataset_list_test.json produced by judge_dataset.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/workspace/llama3b-rm",
        help="Base reward model name or path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rm_model"),
        help="Directory where the trained model checkpoints will be stored.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum number of tokens per preference pair after tokenization.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=2000,
        help="Maximum number of characters kept from each message before tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Batch size per device for the training split.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2,
        help="Batch size per device for the evaluation split.",
    )
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=40,
        help="Frequency (in steps) to evaluate the model when a test split is provided.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=40,
        help="Frequency (in steps) to save checkpoints.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency (in steps) to log training metrics.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=10,
        help="Number of epochs to train the reward model.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name; if omitted, W&B logging is disabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="Optional name for the W&B run.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        help="Optional list of tags to attach to the W&B run.",
    )
    parser.add_argument(
        "--load-in-8bit",
        # action="store_true",
         default=True,
        help="Load the base model in 8-bit using bitsandbytes before applying LoRA adapters.",
    )
    parser.add_argument(
        "--grpo-clip-eps",
        type=float,
        default=0.2,
        help="Clipping epsilon for GRPO (PPO-style ratio clip).",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.0,
        help="KL penalty weight against the reference RMSearch model (0 disables).",
    )
    parser.add_argument(
        "--rm-url",
        type=str,
        default=None,
        help="RMSearch base URL for the reference model used in the KL penalty.",
    )
    parser.add_argument(
        "--rm-timeout",
        type=float,
        default=10.0,
        help="Timeout (seconds) for RMSearch reference calls.",
    )
    args = parser.parse_args()

    if not args.dataset_list_train.exists():
        raise FileNotFoundError(f"Dataset list not found: {args.dataset_list_train}")

    with args.dataset_list_train.open() as handle:
        dataset_list_train = json.load(handle)

    dataset_list_test = None
    if args.dataset_list_test is not None:
        if not args.dataset_list_test.exists():
            raise FileNotFoundError(f"Dataset list not found: {args.dataset_list_test}")
        with args.dataset_list_test.open() as handle:
            dataset_list_test = json.load(handle)

    train_reward_model(
        dataset_list_train,
        dataset_list_test=dataset_list_test,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        max_characters=args.max_characters,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_steps=args.evaluation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        load_in_8bit=args.load_in_8bit,
        grpo_clip_eps=args.grpo_clip_eps,
        kl_coef=args.kl_coef,
        rm_url=args.rm_url,
        rm_timeout=args.rm_timeout,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
