from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path
from typing import Tuple

import tyro
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


@dc.dataclass
class Args:
    
    repo_id:        str             = "JetBrains/LibEvoBench"
    dest_root:      Path            = dc.field(default_factory=lambda: Path("out/benchmark_db"))
    libs:           Tuple[str, ...] = ("torch", "numpy", "scipy")


def main(args: Args) -> None:
    root = args.dest_root.resolve()
    want = set(args.libs)

    try:
        configs = get_dataset_config_names(args.repo_id)
    except Exception as exc:
        sys.exit(f"Cannot list configs: {exc}")

    for cfg in sorted(configs):
        if "__" not in cfg:
            continue
        lib, stem = cfg.split("__", 1)
        if not lib or not stem or lib not in want:
            continue

        out = root / lib / f"{stem}.parquet"
        if out.exists():
            continue

        splits = get_dataset_split_names(args.repo_id, config_name=cfg)
        if not splits:
            continue
        split = splits[0] if len(splits) == 1 else ("train" if "train" in splits else splits[0])

        out.parent.mkdir(parents=True, exist_ok=True)
        load_dataset(args.repo_id, cfg, split=split).to_parquet(out)


if __name__ == "__main__":
    main(tyro.cli(Args))
