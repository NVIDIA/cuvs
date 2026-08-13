from __future__ import annotations
import multiprocessing
import argparse
from pathlib import Path
from .config import Config, NoiseScheme
from .core import run


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser for the scaler entrypoint."""
    p = argparse.ArgumentParser(
        prog="scaler",
        description="Scale FBIN/FVECs datasets by replicating vectors and applying ±noise.",
    )

    p.add_argument("-i", "--input-dataset", required=True, type=Path, dest="input_path",
                   help="Input dataset path (.fbin or .fvecs/.fvec).")
    p.add_argument("-o", "--output-dataset", required=True, type=Path, dest="output_path",
                   help="Output dataset path (.fbin or .fvecs/.fvec).")

    p.add_argument("-s", "--scale", required=True, type=int,
                   help="Number of replicas to generate (>=1).")
    p.add_argument("-b", "--batch-size", default=8192, type=int,
                   help="Vectors per read batch.")
    p.add_argument("-a", "--noise", dest="noise_amplitude", default=0.02, type=float,
                   help="Noise amplitude. Noise is always applied as ±noise.")
    p.add_argument("-m", "--noise-scheme", default="PARTIAL", choices=[e.value for e in NoiseScheme],
                   help="Noise scheme: ALL or PARTIAL.")

    p.add_argument("-n", "--normalize", action="store_true",
                   help="If set: abnormal vectors are renormalized using tolerance.")
    p.add_argument("-t", "--tolerance", default=3.7e-4, type=float,
                   help="Tolerance for normalized vectors: keep vectors with | ||v|| - 1 | <= tolerance.")

    p.add_argument("-k", "--shard-size", default=None, type=int,
                   help="Vectors per shard. If omitted, sharding is applied only if required by int32 FBIN header limit.")
    p.add_argument("-x", "--seed", default=None, type=int,
                   help="Random seed for deterministic noise/dimension selection.")

    p.add_argument("-j", "--json-metadata", default=None, type=Path,
                   help="Optional path to write JSON summary metadata.")
    core_num = multiprocessing.cpu_count() or 1
    p.add_argument(
        "-w",
        "--workers",
        default=core_num,
        type=int,
        help=f"Number of worker threads (>=1). Default: {core_num} (Number of available CPU cores)",
    )


    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse args, run scaling, print a short completion message."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = Config(
        input_path=args.input_path,
        output_path=args.output_path,
        scale=args.scale,
        batch_size=args.batch_size,
        noise_amplitude=args.noise_amplitude,
        noise_scheme=NoiseScheme(args.noise_scheme),
        normalize=bool(args.normalize),
        tolerance=float(args.tolerance),
        shard_size=args.shard_size,
        seed=args.seed,
        json_metadata=args.json_metadata,
        workers=int(args.workers),
    )

    summary = run(cfg)
    print(f"Done. Total vectors written: {summary['total_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
