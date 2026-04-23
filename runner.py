import argparse
import importlib

import train as train_module


DEFAULT_CONFIG = train_module.DEFAULT_CONFIG


def _config_to_dict(config):
    if isinstance(config, dict):
        raw_config = dict(config)
    elif hasattr(config, "__dict__") or hasattr(config, "__class__"):
        raw_config = {}
        for name in dir(config):
            if name.startswith("_"):
                continue
            try:
                value = getattr(config, name)
            except AttributeError:
                continue
            if callable(value):
                continue
            raw_config[name] = value
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")
    return raw_config


def run_training(config):
    # Notebook workflows often edit train.py between runs; reload to avoid stale module state.
    importlib.reload(train_module)
    train_module.train(_config_to_dict(config))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on next-action prediction from HTML and screenshots.")
    parser.add_argument("--model-id", default=DEFAULT_CONFIG["model_id"])
    parser.add_argument("--hf-token", default=DEFAULT_CONFIG["hf_token"])
    parser.add_argument("--hf-cache-dir", default=DEFAULT_CONFIG["hf_cache_dir"])
    parser.add_argument("--image-root", default=DEFAULT_CONFIG["image_root"])
    parser.add_argument("--jsonl-path", default=DEFAULT_CONFIG["jsonl_path"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--epochs", type=float, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--train-bs", type=int, default=DEFAULT_CONFIG["train_bs"])
    parser.add_argument("--eval-bs", type=int, default=DEFAULT_CONFIG["eval_bs"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_CONFIG["grad_accum"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--log-steps", type=int, default=DEFAULT_CONFIG["log_steps"])
    parser.add_argument("--save-steps", type=int, default=DEFAULT_CONFIG["save_steps"])
    parser.add_argument("--use-bf16", action="store_true", default=DEFAULT_CONFIG["use_bf16"])
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--grad-checkpointing", action="store_true", default=DEFAULT_CONFIG["grad_checkpointing"])
    parser.add_argument("--no-grad-checkpointing", action="store_false", dest="grad_checkpointing")
    parser.add_argument("--use-lora", action="store_true", default=DEFAULT_CONFIG["use_lora"])
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_CONFIG["lora_dropout"])
    parser.add_argument("--lora-targets", nargs="*", default=list(DEFAULT_CONFIG["lora_targets"]))
    parser.add_argument("--use-flash-attn", action="store_true", default=DEFAULT_CONFIG["use_flash_attn"])
    parser.add_argument("--no-flash-attn", action="store_false", dest="use_flash_attn")
    parser.add_argument("--use-quantization", action="store_true", default=DEFAULT_CONFIG["use_quantization"])
    parser.add_argument("--no-quantization", action="store_false", dest="use_quantization")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {
        "model_id": args.model_id,
        "hf_token": args.hf_token,
        "hf_cache_dir": args.hf_cache_dir,
        "use_flash_attn": args.use_flash_attn,
        "use_quantization": args.use_quantization,
        "image_root": args.image_root,
        "jsonl_path": args.jsonl_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "grad_accum": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "use_bf16": args.use_bf16,
        "grad_checkpointing": args.grad_checkpointing,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": tuple(args.lora_targets),
    }
    train_module.train(cfg)


if __name__ == "__main__":
    main()
