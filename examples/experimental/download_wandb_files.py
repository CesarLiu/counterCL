import os
import wandb
from eval_utils import (
    load_config,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/experimental/config/eval_config_1000scenes")
    parser.add_argument("--model_config", type=str, default="examples/experimental/config/model_config_1000scenes")
    parser.add_argument("--runs_path", type=str, default="runs/")
    parser.add_argument("--download_models", type=bool, default=True)
    parser.add_argument("--download_curriculum", type=bool, default=True)

    args = parser.parse_args()
    eval_config = load_config(args.config)
    model_config = load_config(args.model_config)
    api = wandb.Api()
    checkpoints = [eval_config.first_ckpt+i*eval_config.every_k_ckpts for i in range(eval_config.number_of_ckpts)]

    for r in model_config.runs:
        print(f"\nRun {r.name} is being procesed...")
        run = api.run(os.path.join(eval_config.wandb_run_path, r.name))
        for file in run.files():
            if str(file).startswith("<File runs/"):
                pt_index = None
                if args.download_models and "model_" in str(file):
                    prefix_index = str(file).index("model_")
                    pt_index = str(file).index(".pt")
                    filename = str(file)[prefix_index:pt_index+3]
                if args.download_curriculum and "curriculum_" in str(file):
                    prefix_index = str(file).index("curriculum_")
                    pt_index = str(file).index(".pkl")
                    filename = str(file)[prefix_index:pt_index+4]
                if pt_index is None:
                    continue
                checkpoint = int(str(file)[pt_index-6:pt_index])
                if checkpoint not in checkpoints:
                    continue
                print(f"File to download: {filename}")
                if os.path.exists(os.path.join(args.runs_path, r.name, filename)):
                    print(f"\tAlready exits. Skipping...")
                    continue
                print("\t------>Downloading...")
                file.download()
                    