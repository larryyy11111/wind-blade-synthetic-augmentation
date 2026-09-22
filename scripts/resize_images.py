"""Resize photographs or class-ID masks into a separate output directory."""
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int, help="Square side length in pixels")
    parser.add_argument("--kind", required=True, choices=["image", "mask"])
    args = parser.parse_args()
    if not args.input.is_dir() or args.size < 1:
        parser.error("Input must be a directory and size must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output directory to preserve original images")
    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        parser.error("No PNG/JPG/JPEG files found")
    targets = [args.output / (p.stem + ".png" if args.kind == "mask" else p.name) for p in files]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        parser.error("Output collision; use an empty directory and unique image stems")
    args.output.mkdir(parents=True, exist_ok=True)
    for source, target in zip(files, targets):
        with Image.open(source) as image:
            mode = Image.Resampling.NEAREST if args.kind == "mask" else Image.Resampling.BILINEAR
            image.resize((args.size, args.size), mode).save(target)
    print(f"Saved {len(files)} files")


if __name__ == "__main__":
    main()
