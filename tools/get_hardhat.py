#!/usr/bin/env python3
"""Загрузка датасета Hard Hat Workers (5000 изображений, разметка helmet/head/person, лицензия CC0)
с Hugging Face (Voxel51/hard-hat-detection) для tools/train_helmet.py.

    python tools/get_hardhat.py --out /data/hardhat
    # через зеркало (Nexus raw/HF-proxy):
    python tools/get_hardhat.py --out /data/hardhat --base http://nexus.local/repository/hf/datasets/Voxel51/hard-hat-detection/resolve/main
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://huggingface.co/datasets/Voxel51/hard-hat-detection/resolve/main"


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    (a.out / "data").mkdir(parents=True, exist_ok=True)
    fetch(f"{a.base}/samples.json", a.out / "samples.json")
    files = [s["filepath"] for s in json.load(open(a.out / "samples.json"))["samples"]]
    done = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for _ in ex.map(lambda fp: fetch(f"{a.base}/{fp}", a.out / fp), files):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(files)}", flush=True)
    print(f"готово: {len(files)} изображений в {a.out}")


if __name__ == "__main__":
    main()
