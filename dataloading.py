import json
import os
from pathlib import Path

from datasets import Dataset


def default_image_root() -> str:
    local_root = os.environ.get("LOCAL")
    if local_root:
        return str(Path(local_root) / "project_data")
    return "./data/images"


def default_jsonl_path() -> str:
    return "./vanilla_matched_6445.jsonl"


def resolve_data_paths(jsonl_path: str | None = None, image_root: str | None = None) -> tuple[str, str]:
    default_jsonl = default_jsonl_path()
    default_image = default_image_root()
    resolved_jsonl = jsonl_path or default_jsonl
    resolved_image_root = image_root or default_image

    # Prefer node-local/current-dir defaults when explicit paths are missing.
    if not Path(resolved_jsonl).exists() and Path(default_jsonl).exists():
        resolved_jsonl = default_jsonl
    if not Path(resolved_image_root).exists() and Path(default_image).exists():
        resolved_image_root = default_image

    # Backward-compatible fallback for older layout under ./data.
    if not Path(resolved_jsonl).exists() and Path("./data/vanilla_matched_6445.jsonl").exists():
        resolved_jsonl = "./data/vanilla_matched_6445.jsonl"
    if not Path(resolved_image_root).exists() and Path("./data/images").exists():
        resolved_image_root = "./data/images"

    return resolved_jsonl, resolved_image_root


def find_image_manifest(image_root: str) -> Path:
    root = Path(image_root)
    manifests = sorted(root.rglob("manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"Could not find manifest.jsonl under {image_root}")
    return manifests[0]


def load_image_lookup(image_root: str) -> dict[str, str]:
    manifest_path = find_image_manifest(image_root)
    lookup: dict[str, str] = {}

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            filename = record.get("filename") or record.get("original_path")
            if not filename:
                continue
            image_path = manifest_path.parent / filename
            for key in (
                record.get("dataset_index"),
                record.get("annotation_id"),
                record.get("source_annotation_id"),
            ):
                if key is not None:
                    lookup[str(key)] = str(image_path)

    if not lookup:
        raise ValueError(f"No image entries found in {manifest_path}")

    return lookup


def load_records(jsonl_path: str | None = None, image_root: str | None = None) -> Dataset:
    resolved_jsonl, resolved_image_root = resolve_data_paths(jsonl_path, image_root)
    path = Path(resolved_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {resolved_jsonl}")

    image_lookup = load_image_lookup(resolved_image_root)

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    examples: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_key = record.get("dataset_index") or record.get("annotation_id") or record.get("source_annotation_id")
        image_path = image_lookup.get(str(record_key)) if record_key is not None else None
        resolved_image_path = None
        if image_path is not None:
            candidate = Path(image_path)
            if not candidate.is_absolute():
                candidate = Path(resolved_image_root) / candidate
            if candidate.exists():
                resolved_image_path = str(candidate)
        system_prompt = record.get("system_prompt") or record.get("system") or ""
        prompt_text = record.get("prompt_text") or record.get("prompt") or record.get("html") or record.get("html_text")
        target_text = record.get("target") or record.get("action") or record.get("target_action") or record.get("label")
        if resolved_image_path is None or not prompt_text or target_text is None:
            continue
        examples.append(
            {
                "image": resolved_image_path,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "target": target_text,
            }
        )

    if not examples:
        raise ValueError(f"No usable records found in {resolved_jsonl}")

    return Dataset.from_list(examples)


def resolve_image_path(image_path: str, image_root: Path) -> str:
    if os.path.isabs(image_path):
        return image_path
    return str(Path(image_root) / image_path)
