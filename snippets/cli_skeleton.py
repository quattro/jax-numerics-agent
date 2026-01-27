import argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args(argv)

    # Call into library code here.
    return run(seed=args.seed, dtype=args.dtype)


def run(*, seed: int, dtype: str) -> int:
    # Placeholder for library call.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
