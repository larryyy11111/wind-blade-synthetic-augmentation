"""Portable wrapper around pytorch-fid, adapted from the uploaded FID/FID.py."""
import argparse
import json
from pathlib import Path
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Optional new JSON result file")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for folder in (args.real, args.fake):
        if not folder.is_dir():
            parser.error(f"Not a directory: {folder}")
    if args.real.resolve() == args.fake.resolve():
        parser.error("Real and fake directories must be different")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new result filename")
    from pytorch_fid.fid_score import IMAGE_EXTENSIONS, calculate_fid_given_paths
    counts = []
    for folder in (args.real, args.fake):
        count = sum(p.is_file() and p.suffix[1:] in IMAGE_EXTENSIONS for p in folder.iterdir())
        if count < 2:
            parser.error(f"Need at least two supported images in {folder}; found {count}")
        counts.append(count)
    if min(counts) <= 2048:
        warnings.warn("Sample count is no larger than the 2048-dimensional feature space; covariance estimates are rank deficient. Treat small-sample FID cautiously.")
    score = float(calculate_fid_given_paths(
        [str(args.real), str(args.fake)], batch_size=args.batch_size,
        device=args.device, dims=2048, num_workers=0))
    import math
    if not math.isfinite(score):
        raise RuntimeError("Non-finite FID; inspect the image sets and numerical stability")
    from importlib.metadata import version
    result = {"fid": score, "real_count": counts[0], "fake_count": counts[1],
              "feature_dimensions": 2048, "device": args.device,
              "batch_size": args.batch_size, "pytorch_fid_version": version("pytorch-fid")}
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
