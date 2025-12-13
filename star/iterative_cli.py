import argparse

from iterative_dpo import iterative_dpo, parse_args as parse_dpo_args
from iterative_sft import iterative_loop, parse_args as parse_sft_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative chess training entrypoint")
    parser.add_argument("--mode", choices=["sft_iter", "dpo_iter", "rlvr"], required=True)
    args, unknown = parser.parse_known_args()

    if args.mode == "sft_iter":
        sft_args = parse_sft_args()
        iterative_loop(sft_args)
    elif args.mode == "dpo_iter":
        dpo_args = parse_dpo_args()
        iterative_dpo(dpo_args)
    else:
        raise NotImplementedError("RLVR mode is not implemented in this reference implementation.")


if __name__ == "__main__":
    main()
