#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path

import gdown


DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/1leHOcLv631qfOniHsI409CW1aC48cEt8/view?usp=sharing"


def normalize_drive_download_url(drive_url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_url)
    if not match:
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", drive_url)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"
    return drive_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download image zip to node-local storage, extract it, and optionally remove the zip."
    )
    parser.add_argument("--drive-url", default=os.environ.get("IMAGE_DRIVE_URL", DEFAULT_DRIVE_URL))
    parser.add_argument("--local-dir", default=os.environ.get("LOCAL"))
    parser.add_argument("--zip-name", default="images.zip")
    parser.add_argument("--extract-dir", default="images")
    parser.add_argument("--keep-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.local_dir:
        raise ValueError("LOCAL is not set. Pass --local-dir or export LOCAL first.")

    local_root = Path(args.local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    zip_path = local_root / args.zip_name
    extract_path = local_root / args.extract_dir
    extract_path.mkdir(parents=True, exist_ok=True)

    direct_url = normalize_drive_download_url(args.drive_url)
    print(f"Downloading: {direct_url}")
    print(f"Zip path: {zip_path}")
    gdown.download(direct_url, str(zip_path), quiet=False)

    print(f"Extracting into: {extract_path}")
    subprocess.run(["unzip", "-qo", str(zip_path), "-d", str(extract_path)], check=True)

    if args.keep_zip:
        print(f"Keeping zip file: {zip_path}")
    else:
        zip_path.unlink(missing_ok=True)
        print(f"Removed zip file: {zip_path}")

    print("Done.")


if __name__ == "__main__":
    main()