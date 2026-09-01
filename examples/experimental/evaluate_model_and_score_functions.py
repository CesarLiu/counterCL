import torch
import pandas as pd
from box import Box
import numpy as np
import os
import logging
from gpudrive.env.dataset import SceneDataLoader
from eval_utils import (
    load_config,
    make_env,
    load_policy,
    evaluate_policy_with_scores,
)

import random
import torch
import numpy as np
import wandb

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)
SEED = 42  # Set to any fixed value
set_seed(SEED)

def download_run_from_wandb(run, runs_path,wandb_run_path, first_ckpt, every_k_ckpts, number_of_ckpts):
    available_ckpts = []
    available_models = []
    available_files = []
    api = wandb.Api()
    run = api.run(os.path.join(wandb_run_path, run.name))
    for file in run.files():
        if str(file).startswith("<File runs/") and "model_" in str(file):
            prefix_index = str(file).index("model_")
            pt_index = str(file).index(".pt")
            filename = str(file)[prefix_index:pt_index+3]
            checkpoint = int(str(file)[pt_index-6:pt_index])
            available_ckpts.append(checkpoint)
            available_models.append(file)
            available_files.append(filename)
    # Sort by checkpoint
    # first_ckpt = min(available_ckpts)
    count_ckpts = 0
    for checkpoint, model, filename in zip(available_ckpts, available_models, available_files):
        if checkpoint >= first_ckpt and (checkpoint - first_ckpt) % every_k_ckpts == 0:
            count_ckpts += 1
            if os.path.exists(os.path.join(runs_path, run.name, filename)):
                print(f"---> File already exists: {filename}")
                continue
            print(f"---> Downloading file: {filename} from wandb...")
            model.download()
            if count_ckpts >= number_of_ckpts:
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/experimental/config/eval_config_default_1000scenes_computeab")
    parser.add_argument("--model_config", type=str, default="examples/experimental/config/model_config_default_1000scenes_computeab")
    parser.add_argument("--res_path", type=str)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--valid_dir", type=str)
    parser.add_argument("--train_dataset_size", type=int)
    parser.add_argument("--test_dataset_size", type=int)
    parser.add_argument("--valid_dataset_size", type=int)
    parser.add_argument("--load_existing_df", type=bool, default=True)
    parser.add_argument("--eval_train_dataset", type=bool)
    parser.add_argument("--eval_test_dataset", type=bool)
    parser.add_argument("--eval_valid_dataset", type=bool)
    parser.add_argument("--runs_path", type=str, default="runs/")
    parser.add_argument("--wandb_run_path", type=str)
    args = parser.parse_args()
    # Load configurations
    eval_config = load_config(args.config)
    model_config = load_config(args.model_config)

    passed_args = {
        "res_path": args.res_path,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "valid_dir": args.valid_dir,
        "train_dataset_size": args.train_dataset_size,
        "test_dataset_size": args.test_dataset_size,
        "valid_dataset_size": args.valid_dataset_size,
        "eval_train_dataset": args.eval_train_dataset,
        "eval_test_dataset": args.eval_test_dataset,
        "eval_valid_dataset": args.eval_valid_dataset,
        "wandb_run_path": args.wandb_run_path,
    }
    eval_config.update(
        {k: v for k, v in passed_args.items() if v is not None}
    )
    
    # Make directory to save results
    if not os.path.exists(eval_config.res_path):
        os.makedirs(eval_config.res_path, exist_ok=True)

    # Make environment
    env = make_env(eval_config, 
                   SceneDataLoader(
                       root=eval_config.train_dir,
                       batch_size=eval_config.num_worlds,
                       dataset_size=eval_config.num_worlds,
                       sample_with_replacement=False,))

    for run in model_config.runs:
        print(f"\n\nRun to process: {run.name}")
        download_run_from_wandb(run, args.runs_path, eval_config.wandb_run_path, eval_config.first_ckpt, eval_config.every_k_ckpts, eval_config.number_of_ckpts)
        curriculum_type = run.curriculum_type
        train_seed = run.train_seed
        score_type, beta, rho = None, None, None
        df_res_filename = f"run={run.name}_cur={curriculum_type}"
        if curriculum_type == "plr":
            score_type, beta, rho = run.score_type, run.beta, run.rho
            df_res_filename += f"_score={score_type}_beta={beta}_rho={rho}"
        df_res_filename += f"_seed={train_seed}_label={run.label}.csv"
        df_res_run = None
        loaded_existing_df = False
        if args.load_existing_df and os.path.exists(os.path.join(eval_config.res_path, df_res_filename)):
            print("Found a df! Loading...")
            df_res_run = pd.read_csv(os.path.join(eval_config.res_path, df_res_filename))
            loaded_existing_df = True
        run_models_path = os.path.join(args.runs_path, run.name)
        checkpoints = []
        for model_name in os.listdir(run_models_path):
            if model_name[-3:] != ".pt":
                continue
            checkpoints.append(int(model_name[-9:-3]))
        checkpoints.sort()
        # Only use every_k_ckpts checkpoints from the first checkpoint onwards
        print(f"Existing models at checkpoints {checkpoints}")
        checkpoints = [ch for ch in checkpoints if ch >= eval_config.first_ckpt]
        checkpoints = [ch for ch in checkpoints if (ch - eval_config.first_ckpt) % eval_config.every_k_ckpts == 0]
        checkpoints = checkpoints[:eval_config.number_of_ckpts]
        print(f"Using checkpoints {checkpoints}")
        for checkpoint in checkpoints:
            if loaded_existing_df and (df_res_run["checkpoint"] == checkpoint).any():
                print(f"Checkpoint {checkpoint} exists in df. Skipping...")
                continue
            chkpt_zeros = ""
            for z in range(6 - len(str(checkpoint))):
                chkpt_zeros += "0"
            model_name = f"model_{run.name}_{chkpt_zeros}{checkpoint}.pt"
            
            logging.info(f"Evaluating curriculum={curriculum_type}, score={score_type}, beta={beta}, rho={rho}, seed={train_seed}, label={run.label} at checkpoint {checkpoint}...")

            # Load policy
            policy = load_policy(
                path_to_cpt=run_models_path,
                model_name=model_name[:-3],
                device=eval_config.device,
                env=env
            )

            df_res_ckpt = None
            evaluated_datasets = []
            if eval_config.eval_train_dataset:
                train_loader = SceneDataLoader(
                    root=eval_config.train_dir,
                    batch_size=eval_config.num_worlds,
                    dataset_size=run.train_dataset_size
                    if eval_config.deterministic
                    else run.train_dataset_size*eval_config.num_copies,
                    sample_with_replacement=False,
                    shuffle=False,
                )
                logging.info(f"Rollouts on {len(set(train_loader.dataset))} train scenes")
                df_res_train_ckpt = evaluate_policy_with_scores(
                    env=env,
                    policy=policy,
                    data_loader=train_loader,
                    dataset_name="train",
                    deterministic=eval_config.deterministic,
                    render_sim_state=False,
                    gamma=eval_config.gamma,
                    lambd=eval_config.lambd,
                )
                df_res_ckpt = df_res_train_ckpt
                evaluated_datasets.append("train")

            if eval_config.eval_test_dataset:
                test_loader = SceneDataLoader(
                    root=eval_config.test_dir,
                    batch_size=eval_config.num_worlds,
                    dataset_size=eval_config.test_dataset_size
                    if eval_config.deterministic
                    else eval_config.test_dataset_size*eval_config.num_copies,
                    sample_with_replacement=False,
                    shuffle=False,
                )
                logging.info(f"Rollouts on {len(set(test_loader.dataset))} test scenes")
                df_res_test_ckpt = evaluate_policy_with_scores(
                    env=env,
                    policy=policy,
                    data_loader=test_loader,
                    dataset_name="test",
                    deterministic=eval_config.deterministic,
                    render_sim_state=False,
                    gamma=eval_config.gamma,
                    lambd=eval_config.lambd,
                )
                if df_res_ckpt is None:
                    df_res_ckpt = df_res_test_ckpt
                else:
                    df_res_ckpt = pd.concat([df_res_ckpt, df_res_test_ckpt])
                evaluated_datasets.append("test")

            if eval_config.eval_valid_dataset:
                valid_loader = SceneDataLoader(
                    root=eval_config.valid_dir,
                    batch_size=eval_config.num_worlds,
                    dataset_size=eval_config.valid_dataset_size
                    if eval_config.deterministic
                    else eval_config.valid_dataset_size*eval_config.num_copies,
                    sample_with_replacement=False,
                    shuffle=False,
                )
                logging.info(f"Rollouts on {len(set(valid_loader.dataset))} valid scenes")
                df_res_valid_ckpt = evaluate_policy_with_scores(
                    env=env,
                    policy=policy,
                    data_loader=valid_loader,
                    dataset_name="valid",
                    deterministic=eval_config.deterministic,
                    render_sim_state=False,
                    gamma=eval_config.gamma,
                    lambd=eval_config.lambd,
                )
                if df_res_ckpt is None:
                    df_res_ckpt = df_res_valid_ckpt
                else:
                    df_res_ckpt = pd.concat([df_res_ckpt, df_res_valid_ckpt])
                evaluated_datasets.append("valid")

            # Add metadata
            df_res_ckpt["model_name"] = run.name
            df_res_ckpt["curriculum_type"] = curriculum_type
            df_res_ckpt["score_type"] = score_type
            df_res_ckpt["beta"] = beta
            df_res_ckpt["rho"] = rho
            df_res_ckpt["train_seed"] = train_seed
            df_res_ckpt["checkpoint"] = checkpoint

            print("Scene-based metrics \n")
            tab_agg_perf = df_res_ckpt.groupby("dataset")[
                [
                    "goal_achieved_frac",
                    "collided_frac",
                    "off_road_frac",
                    "other_frac",
                ]
            ].agg(["mean", "std"])
            tab_agg_perf = tab_agg_perf * 100
            tab_agg_perf = tab_agg_perf.round(1)
            print(tab_agg_perf)

            if df_res_run is None:
                df_res_run = df_res_ckpt
            else:
                df_res_run = pd.concat([df_res_run, df_res_ckpt], ignore_index=True)

            df_res_run.to_csv(f"{eval_config.res_path}/{df_res_filename}", index=False)
            logging.info(f"Saved at {eval_config.res_path}/{df_res_filename}\n")

        print("\nScene-based metrics \n")
        tab_agg_perf = df_res_run.groupby(["dataset", "checkpoint"])[
            [
                "goal_achieved_frac",
                "collided_frac",
                "off_road_frac",
                "other_frac",
            ]
        ].agg(["mean", "std"])
        tab_agg_perf = tab_agg_perf * 100
        tab_agg_perf = tab_agg_perf.round(1)
        print(tab_agg_perf)

        print("\nAgent-based metrics \n")
        for ds in evaluated_datasets:
            print_total_agents = False
            for checkpoint in checkpoints:
                ckpt_idx = (df_res_run["checkpoint"]==checkpoint) & (df_res_run["dataset"]==ds)
                if not ckpt_idx.any():
                    print(f"No data for checkpoint {checkpoint} in dataset {ds}")
                    continue
                df_res_ckpt_ds = df_res_run[ckpt_idx]
                total_agents = df_res_ckpt_ds["controlled_agents_in_scene"].sum()
                collision_rate = (df_res_ckpt_ds["collided_count"].sum() / total_agents) * 100
                offroad_rate = (df_res_ckpt_ds["off_road_count"].sum() / total_agents) * 100
                goal_rate = (df_res_ckpt_ds["goal_achieved_count"].sum() / total_agents) * 100
                other_rate = (df_res_ckpt_ds["other_count"].sum() / total_agents) * 100
                if not print_total_agents:
                    print(f"Dataset: {ds} - Total agents: {total_agents} in {df_res_ckpt_ds.shape[0]} scenes")
                    print_total_agents = True
                print(f"\t\tCheckpoint: {checkpoint:04d}, goal:{goal_rate:03.2f}%, coll:{collision_rate:03.2f}%, offr:{offroad_rate:03.2f}%")
