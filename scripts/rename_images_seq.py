from __future__ import annotations

import argparse
from pathlib import Path


def _collect_files(root: Path, exts: list[str]) -> list[Path]:
    files = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        files.append(p)
    return sorted(files, key=lambda x: x.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename images to 1.jpg, 2.jpg, ...")
    parser.add_argument("dir", help="Directory containing images to rename.")
    parser.add_argument("--start", type=int, default=1, help="Start index (default: 1).")
    parser.add_argument("--ext", default="jpg", help="Extension to filter (default: jpg).")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned changes.")
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    ext = args.ext.lower().lstrip(".")
    exts = [f".{ext}"] if ext else []
    files = _collect_files(root, exts)
    if not files:
        print("No files found.")
        return

    # Stage 1: rename to temp to avoid collisions
    temp_paths: list[Path] = []
    for i, p in enumerate(files, start=args.start):
        tmp = root / f"__tmp__{i}{p.suffix.lower()}"
        if args.dry_run:
            print(f"{p.name} -> {tmp.name}")
        else:
            p.rename(tmp)
        temp_paths.append(tmp)

    # Stage 2: rename to final names
    for i, p in enumerate(temp_paths, start=args.start):
        final = root / f"{i}{exts[0] if exts else p.suffix.lower()}"
        if args.dry_run:
            print(f"{p.name} -> {final.name}")
        else:
            p.rename(final)

    if not args.dry_run:
        print(f"Renamed {len(files)} files in {root}")


if __name__ == "__main__":
    main()
