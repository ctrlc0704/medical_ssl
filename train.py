import argparse
import yaml
from pretrain import run_pretrain
from finetune import run_finetune

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"], required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.stage == "pretrain":
        run_pretrain(cfg)
    else:
        run_finetune(cfg)

if __name__ == "__main__":
    main()