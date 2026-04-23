import os
import gc
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments, Qwen3VLForConditionalGeneration

from dataloading import load_records, resolve_image_path


LOCAL_ROOT = os.environ.get("LOCAL")
DEFAULT_IMAGE_ROOT = str(Path(LOCAL_ROOT) / "project_data") if LOCAL_ROOT else "./data/images"


DEFAULT_CONFIG = {
    "model_id": "Qwen/Qwen3-VL-8B-Thinking",
    "hf_token": None,
    "hf_cache_dir": "./.hf_cache",
    "use_flash_attn": True,
    "use_quantization": False,
    "image_root": DEFAULT_IMAGE_ROOT,
    "jsonl_path": "./vanilla_matched_6445.jsonl",
    "output_dir": "./outputs/qwen-vl-next-action-sft",
    "max_length": 2048,
    "epochs": 1,
    "lr": 1e-5,
    "train_bs": 1,
    "eval_bs": 1,
    "grad_accum": 4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "log_steps": 10,
    "save_steps": 200,
    "use_bf16": True,
    "grad_checkpointing": True,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"),
    "max_prompt_chars": 800,
    "image_max_side": 320,
    "dataloader_num_workers": 0,
    "cache_clear_steps": 50,
}


def build_messages(system_prompt: str, prompt_text: str, target_text: str | None = None):
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(user_msg)
    if target_text is None:
        return messages
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": target_text.strip()}]}
    return messages + [assistant_msg]


class VLMActionCollator:
    def __init__(
        self,
        processor,
        image_root: Path,
        max_length: int = 2048,
        max_prompt_chars: int | None = None,
        image_max_side: int | None = None,
    ):
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length
        self.max_prompt_chars = max_prompt_chars
        self.image_max_side = image_max_side

    def __call__(self, features):
        images = []
        full_texts = []
        prompt_texts = []

        for example in features:
            image = Image.open(resolve_image_path(example["image"], self.image_root)).convert("RGB")
            if self.image_max_side and max(image.size) > self.image_max_side:
                resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                image.thumbnail((self.image_max_side, self.image_max_side), resampling)
            images.append(image)

            prompt_text = example["prompt_text"]
            if self.max_prompt_chars and len(prompt_text) > self.max_prompt_chars:
                prompt_text = prompt_text[: self.max_prompt_chars]

            full_msgs = build_messages(example.get("system_prompt", ""), prompt_text, example["target"])
            prompt_msgs = build_messages(example.get("system_prompt", ""), prompt_text, None)

            full_texts.append(
                self.processor.apply_chat_template(
                    full_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        full_batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        labels = full_batch["input_ids"].clone()
        for index in range(labels.size(0)):
            prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
            labels[index, :prompt_length] = -100

        full_batch["labels"] = labels
        return full_batch


class CUDACacheCleanupCallback(TrainerCallback):
    def __init__(self, every_n_steps: int):
        self.every_n_steps = max(0, int(every_n_steps))

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps <= 0:
            return control
        if torch.cuda.is_available() and state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return control


def build_model_and_processor(config: dict):
    print("Loading model...")
    attn_impl = "flash_attention_2" if config["use_flash_attn"] else "sdpa"
    token = config.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    cache_dir = config.get("hf_cache_dir")

    if config.get("use_quantization", False):
        raise ValueError("Quantization is disabled for H100 training; use bf16 instead.")

    dtype = torch.bfloat16 if config["use_bf16"] and torch.cuda.is_available() else torch.float16
    model_id = config["model_id"]
    print(f"Trying model_id={model_id}")

    try:
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                trust_remote_code=True,
                token=token,
            )
            print(f"Model loaded with attn_implementation={attn_impl}")
        except Exception as exc:
            if attn_impl != "sdpa":
                print(f"Failed with {attn_impl}: {exc}")
                print("Falling back to sdpa...")
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    token=token,
                )
                print("Model loaded with attn_implementation=sdpa")
            else:
                raise

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model_id={model_id}. If your model is private/gated, set HF_TOKEN/HUGGINGFACE_HUB_TOKEN "
            "or run 'hf auth login'. You can also set MODEL_ID to a local checkpoint path."
        ) from exc

    if config["grad_checkpointing"]:
        model.gradient_checkpointing_enable()

    if config["use_lora"]:
        peft_cfg = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            target_modules=list(config["lora_targets"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model, processor


def train(config: dict):
    config = {**DEFAULT_CONFIG, **config}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    hf_cache_dir = os.path.abspath(config["hf_cache_dir"])
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")

    os.makedirs(config["output_dir"], exist_ok=True)
    image_root = Path(config["image_root"])
    dataset = load_records(config["jsonl_path"], str(image_root))
    print(dataset)
    print("Columns:", dataset.column_names)

    split = dataset.train_test_split(test_size=0.02, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print("Train size:", len(train_ds))
    print("Eval size:", len(eval_ds))
    print(
        "Memory settings:",
        {
            "train_bs": config["train_bs"],
            "grad_accum": config["grad_accum"],
            "effective_batch": config["train_bs"] * config["grad_accum"],
            "max_prompt_chars": config.get("max_prompt_chars"),
            "image_max_side": config.get("image_max_side"),
            "cache_clear_steps": config.get("cache_clear_steps"),
        },
    )

    model, processor = build_model_and_processor(config)
    data_collator = VLMActionCollator(
        processor=processor,
        image_root=image_root,
        max_length=config["max_length"],
        max_prompt_chars=config.get("max_prompt_chars"),
        image_max_side=config.get("image_max_side"),
    )

    args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        per_device_train_batch_size=config["train_bs"],
        per_device_eval_batch_size=config["eval_bs"],
        gradient_accumulation_steps=config["grad_accum"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["log_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["save_steps"],
        save_total_limit=2,
        bf16=config["use_bf16"] and torch.cuda.is_available(),
        fp16=(not config["use_bf16"]) and torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        gradient_checkpointing=config["grad_checkpointing"],
    )

    callbacks = []
    if config.get("cache_clear_steps", 0):
        callbacks.append(CUDACacheCleanupCallback(config["cache_clear_steps"]))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])
    print("Saved to", config["output_dir"])
