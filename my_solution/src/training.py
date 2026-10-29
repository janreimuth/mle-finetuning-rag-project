"""Local LoRA fine-tuning with Hugging Face PEFT.

Trains a LoRA adapter on data/instruction_examples.jsonl and saves the adapter,
tokenizer and a training_summary.json describing the run.

    uv add torch transformers peft
    uv run python src/training.py
"""

import hashlib
import json
import os
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root, next to my_solution/
DATA_PATH = PROJECT_ROOT / "data" / "training" / "instruction_examples.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "my_solution" / "outputs" / "lora-support-adapter"

# Base model, pinned to an exact commit
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DEVICE = torch.device("cpu")

# LoRA
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Optimization
MAX_LENGTH = 512        # prompts now include retrieved context, so 128 is too short
EPOCHS = 6
BATCH_SIZE = 2
LEARNING_RATE = 2e-3
SEED = 7

SYSTEM_MESSAGE = (
    "You are a concise support assistant. Answer only from the context "
    "and cite the sources you used."
)


def set_seed(seed: int) -> None:
    # Keep the small training run reproducible across reruns.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- prompt format
# Native chat template. Inference must use build_inference_prompt so the
# assistant-turn prefix is identical to training.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
tokenizer.pad_token = tokenizer.eos_token


def build_instruction(context: str, question: str) -> str:
    return f"Context:\n{context.strip()}\n\nQuestion: {question.strip()}"


def build_response(answer: str, reference_ids: list[str]) -> str:
    return f"{answer.strip()}\nSources: {', '.join(reference_ids)}"


def build_inference_prompt(instruction: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_training_pair(record: dict) -> tuple[str, str]:
    instruction = build_instruction(record["context"], record["question"])
    response = build_response(record["answer"], record["reference_ids"])
    if not record["answer"].strip():
        raise ValueError(f"{record['id']}: training answers must not be empty.")

    prompt = build_inference_prompt(instruction)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt):
        raise ValueError("The training template does not preserve the inference prefix.")
    # Keep the template's end-of-turn token; do not append a second EOS.
    return prompt, full_text[len(prompt):]


# ---------------------------------------------------------------- dataset

class InstructionDataset(Dataset):
    """Tokenized tensors with labels on the response tokens only."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prompt, response = self.pairs[index]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        token_ids = tokenizer.encode(prompt + response, add_special_tokens=False)
        if token_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Example {index}: tokenization changed the prompt/answer boundary.")
        if len(token_ids) <= len(prompt_ids) or len(token_ids) > MAX_LENGTH:
            raise ValueError(
                f"Example {index}: empty or overlong answer ({len(token_ids)} tokens); "
                "review the example or increase MAX_LENGTH."
            )
        padding = MAX_LENGTH - len(token_ids)
        input_ids = torch.tensor(token_ids + [tokenizer.pad_token_id] * padding)
        attention_mask = torch.tensor([1] * len(token_ids) + [0] * padding)
        labels = input_ids.clone()
        # Mask prompt and padding, but keep the genuine end-of-answer token.
        labels[: len(prompt_ids)] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------- model

def build_model():
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    base_model.config.pad_token_id = tokenizer.eos_token_id
    base_model.config.use_cache = False
    base_model.loss_type = "ForCausalLM"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)
    return model


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def check_only_lora_trainable(model: torch.nn.Module) -> None:
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any("lora_" not in n for n in names):
        raise RuntimeError(f"Unexpected trainable parameters: {names[:5]}")


# ---------------------------------------------------------------- optimization loop

def train_loop(model, train_loader) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    model.train()
    loss_history: list[float] = []
    epoch_losses: list[float] = []

    for epoch in range(EPOCHS):
        current: list[float] = []
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; inspect the token labels.")
            loss.backward()                       # gradients reach only LoRA weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach().cpu())
            loss_history.append(value)
            current.append(value)

        mean_loss = sum(current) / len(current)
        epoch_losses.append(round(mean_loss, 3))
        print(f"Epoch {epoch + 1}/{EPOCHS}: mean loss = {mean_loss:.3f}")

    model.eval()
    return loss_history, epoch_losses


def generate_reply(model, instruction: str, max_new_tokens: int = 64) -> str:
    # Decode only the generated assistant turn, not the input conversation.
    prompt = build_inference_prompt(instruction)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------- main

def train() -> None:
    set_seed(SEED)

    # Keep the exact input bytes for the adapter's training-data fingerprint.
    training_bytes = DATA_PATH.read_bytes()
    records = [json.loads(line) for line in training_bytes.splitlines() if line.strip()]
    pairs = [format_training_pair(r) for r in records]

    dataset = InstructionDataset(pairs)
    for i in range(len(dataset)):          # validate every example before training
        dataset[i]
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = build_model()
    check_only_lora_trainable(model)
    trainable, total = count_parameters(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    loss_history, epoch_losses = train_loop(model, train_loader)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].revision = MODEL_REVISION
    # Only query/value adapters were trained; token embeddings are unchanged.
    model.save_pretrained(str(OUTPUT_DIR), save_embedding_layers=False)
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    summary = {
        "base_model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "training_data": str(DATA_PATH),
        "training_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "training_examples": len(records),
        "prompt_format": {
            "type": "native-chat",
            "system_message": SYSTEM_MESSAGE,
            "user_turn": "Context:\\n{context}\\n\\nQuestion: {question}",
            "assistant_turn": "{answer}\\nSources: {reference_ids}",
            "loss": "assistant tokens only, including end-of-turn token",
        },
        "lora": {
            "target_modules": LORA_TARGET_MODULES,
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
        },
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW, constant learning rate",
        "seed": SEED,
        "device": str(DEVICE),
        "parameters": {
            "trainable": trainable,
            "total": total,
            "trainable_percent": round(100 * trainable / total, 4),
        },
        "epoch_losses": epoch_losses,
        "loss_history": [round(v, 4) for v in loss_history],
        "adapter_output_path": str(OUTPUT_DIR.resolve()),
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Adapter saved to {OUTPUT_DIR.resolve()}")
    print(sorted(p.name for p in OUTPUT_DIR.iterdir()))

    # Smoke test on the first training example
    sample = records[0]
    print("\nSample reply:")
    print(generate_reply(model, build_instruction(sample["context"], sample["question"])))

