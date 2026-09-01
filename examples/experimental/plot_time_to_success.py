import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from eval_utils import (
    load_config,
)

def method_to_label(method):
    if "LEARN-GOAL" in method:
        return "PLR+Learn"
    elif "LEARN" in method:
        return "PLR+Learn-Hard"
    elif "ACT-MAE" in method:
        return "PLR+Act-MAE"
    elif "GC" in method:
        return "PLR+GC-ADE"
    elif "MAX" in method:
        return "PLR+MaxMC"
    elif "PVL" in method:
        return "PLR+PVL"
    else:
        return method

def load_all_dfs(runs, df_path, checkpoints_to_expect, label_with_params=False):
    df_all = None
    for run in runs:
        df_filename = f"run={run.name}_cur={run.curriculum_type}"
        if run.curriculum_type == "plr":
            df_filename += f"_score={run.score_type}_beta={run.beta}_rho={run.rho}"
        df_filename += f"_seed={run.train_seed}_label={run.label}.csv"
        if not os.path.exists(os.path.join(df_path, df_filename)):
            print(f"\nFile does not exist: {df_filename}")
            continue
        df_run = pd.read_csv(os.path.join(df_path, df_filename))
        df_run_columns = df_run.columns.tolist()
        if "rho" not in df_run_columns:
            beta_idx = df_run_columns.index("beta")
            df_run_columns_new = df_run_columns[:beta_idx+1] + ["rho"] + df_run_columns[beta_idx+1:] 
            df_run["rho"] = run.rho if run.curriculum_type == "plr" else None
            df_run = df_run[df_run_columns_new]
        method = f"{run.curriculum_type.upper()}"
        if run.curriculum_type == "plr":
            method += f"+{run.score_type.upper()}"
        if label_with_params and run.curriculum_type == "plr":
            method += f"-beta={run.beta}-rho={run.rho}"
        df_run["method"] = method
        ckpts = df_run['checkpoint'].unique()
        ckpts_to_drop = [ckpt for ckpt in ckpts if ckpt not in checkpoints_to_expect]
        ckpts_to_drop.sort()
        for ckpt_to_drop in ckpts_to_drop:
            df_run.drop(df_run[df_run['checkpoint']==ckpt_to_drop].index, inplace=True)
        if df_all is None:
            df_all = df_run
        else:
            df_all = pd.concat([df_all, df_run], ignore_index=True)
    return df_all

def print_aggragated_performance(df_all):
    tab_agg_perf = df_all.groupby(["dataset", "checkpoint", "method"])[
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

def get_stats_to_reach(df, df_wandb_dict, percentage, checkpoints_to_expect):
    checkpoint_to_reach = None
    uptime_to_reach = None
    step_to_reach = None
    if len(df[df['mean'] >= percentage]['checkpoint'].values) > 0:
        checkpoint_to_reach = df[df['mean'] >= percentage]['checkpoint'].values[0]
        print(f"Checkpoint to reach {percentage}%: {checkpoint_to_reach}")    
        checkpoint_index = checkpoints_to_expect.index(checkpoint_to_reach)
        uptime_to_reach = df_wandb_dict["uptime"][checkpoint_index].mean().item()
        step_to_reach = df_wandb_dict["step"][checkpoint_index].mean().item()
        print(f"First checkpoint to reach {percentage}%: {checkpoint_to_reach}")
    return checkpoint_to_reach, step_to_reach, uptime_to_reach

def get_method_order(methods_stats, checkpoints_to_expect):
    # First sort by the first index of checkpoints_to_reach
    # Then sort by the second index of checkpoints_to_reach
    # Then sort by the third index of checkpoints_to_reach
    method_order = [method for method in methods_stats]
    min_final_checkpoint = min(len(methods_stats[method]["checkpoints_to_reach"]) for method in method_order)
    for i in reversed(range(min_final_checkpoint)):
        method_order = sorted(method_order, key=lambda x: methods_stats[x]["checkpoints_to_reach"][i])
    return method_order


def plot_time_to_success(df, df_wandb_dict, percentages, checkpoints_to_expect, batch_size, colormap_path, figname=None):
    print(df_wandb_dict.keys())
    df_long = df.melt(
        id_vars=['method', 'checkpoint', 'scene', 'train_seed'],
        value_vars=['goal_achieved_frac'],
        var_name='goal_achieved_frac',
        value_name='value'
    )
    # Step 1: average over seeds per scene
    per_scene = df_long.groupby(
        ['method', 'checkpoint', 'train_seed']
    )['value'].mean().reset_index()

    # Step 2: compute mean and std over scenes
    agg = per_scene.groupby(
        ['method', 'checkpoint']
    )['value'].agg(['mean', 'std']).reset_index()
    
    # Prepare x-axis mapping and offsets
    checkpoints = sorted(agg['checkpoint'].unique())
    checkpoint_pos = {cp: i for i, cp in enumerate(checkpoints)}
    methods = sorted(agg['method'].unique())
    n_methods = len(methods)
    offsets = np.zeros((n_methods))  # or np.linspace(-0.2, 0.2, n_methods)
    palette = sns.color_palette(n_colors=n_methods)
    if os.path.exists(colormap_path):
        import json
        with open(colormap_path, 'r') as f:
            color_map = json.load(f)
        print("Using provided color map")
    else:
        color_map = {m: palette[i] for i, m in enumerate(methods)}
    offsets = np.zeros((n_methods))
    for i in range(n_methods):
        offsets[i] = i
    matplotlib.rcParams.update({'font.size': 20})
    # fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    width = 0.9 / n_methods  # the width of the bars
    max_length = 0
    methods_stats = {}
    for i, method in enumerate(methods):
        print(f"method: {method}")
        mdf = agg[agg['method'] == method]
        checkpoints_to_reach = []
        uptimes_to_reach = []
        steps_to_reach = []
        for percentage in percentages:
            checkpoint_to_reach, step_to_reach, uptime_to_reach = get_stats_to_reach(mdf, df_wandb_dict[method], percentage, checkpoints_to_expect)
            if checkpoint_to_reach is not None:
                checkpoints_to_reach.append(checkpoint_to_reach)
                steps_to_reach.append(step_to_reach)
                uptimes_to_reach.append(uptime_to_reach)
        max_length = max(max_length, len(checkpoints_to_reach))
        methods_stats[method] = {
            "checkpoints_to_reach": checkpoints_to_reach,
            "steps_to_reach": steps_to_reach,
            "uptimes_to_reach": uptimes_to_reach
        }
    method_order = get_method_order(methods_stats, checkpoints_to_expect)
    for i, method in enumerate(method_order):
        checkpoints_to_reach = methods_stats[method]["checkpoints_to_reach"]
        steps_to_reach = methods_stats[method]["steps_to_reach"]
        uptimes_to_reach = methods_stats[method]["uptimes_to_reach"]
        ax.bar(np.arange(len(checkpoints_to_reach))+offsets[i]*width, checkpoints_to_reach, color=color_map[method], label=method_to_label(method), width=width, align='center')
        forward = lambda x: x*batch_size
        inverse = lambda x: x/batch_size
        ax2 = ax.secondary_yaxis('right', functions=(forward, inverse))

    # ax.set_xticks(np.arange(len(checkpoints_to_reach)) + (len(checkpoints_to_reach)-1)*width/2)
    ax.set_xticks(np.arange(len(checkpoints_to_reach)) + n_methods*width/2 - width/2)
    ax.set_xticklabels([f"{int(percentages[i]*100)}%" for i in range(len(checkpoints_to_reach))])
    ax.set_xlabel("Success rate")
    ax.set_ylabel(r"Number of policy updates $\downarrow$")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.set_ylabel(r"Number of interactions $\downarrow$")
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend(title="Method", loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.05))
    plt.savefig(figname, dpi=500, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def process_wandb_log(df_wandb, model_config, checkpoints):
    checkpoints = [ch-1 for ch in checkpoints]
    df_wandb_dict = defaultdict(dict)
    for run in model_config.runs:
        method_key = f"{run.curriculum_type.upper()}"
        checkpoints_that_exist = [ckpt for ckpt in checkpoints if ckpt < max(df_wandb[f"performance/epoch"].values.tolist())]
        if run.curriculum_type == "plr":
            method_key += f"+{run.score_type.upper()}"
        step = np.array(df_wandb[f"{run.name} - _step"].values.tolist())[checkpoints_that_exist].reshape(-1, 1)
        uptime = np.array(df_wandb[f"{run.name} - performance/uptime"].values.tolist())[checkpoints_that_exist].reshape(-1, 1)/3600
        if "step" not in df_wandb_dict[method_key]:
            df_wandb_dict[method_key]["step"] = step
            df_wandb_dict[method_key]["uptime"] = uptime
        else:
            min_length = min(df_wandb_dict[method_key]["step"].shape[0], step.shape[0])
            df_wandb_dict[method_key]["step"] = np.concatenate([df_wandb_dict[method_key]["step"][:min_length], step[:min_length]], axis=1)
            df_wandb_dict[method_key]["uptime"] = np.concatenate([df_wandb_dict[method_key]["uptime"][:min_length], uptime[:min_length]], axis=1)
    return df_wandb_dict

if __name__ == "__main__":
    # Load configurations
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/experimental/config/eval_config_step3")
    parser.add_argument("--model_config", type=str, default="examples/experimental/config/model_config_step3_plot")
    parser.add_argument("--colormap_path", type=str, default="color_map.json")
    parser.add_argument("--wandb_log_file", type=str)
    parser.add_argument("--res_path", type=str)
    parser.add_argument("--figure_dir", type=str)
    parser.add_argument("--figname_postfix", type=str)
    args = parser.parse_args()

    percentages = [0.95, 0.96, 0.97, 0.98, 0.99]

    eval_config = load_config(args.config)
    model_config = load_config(args.model_config)

    passed_args = {
        "wandb_log_file": args.wandb_log_file,
        "res_path": args.res_path,
        "figure_dir": args.figure_dir,
        "figname_postfix": args.figname_postfix,
    }
    eval_config.update(
        {k: v for k, v in passed_args.items() if v is not None}
    )

    wandb_log_file = eval_config.wandb_log_file
    res_path = eval_config.res_path
    figure_dir = eval_config.figure_dir
    figname_postfix = eval_config.figname_postfix
    batch_size = int(eval_config.batch_size)
    checkpoints_to_expect = [eval_config.first_ckpt+i*eval_config.every_k_ckpts for i in range(eval_config.number_of_ckpts)]
    label_with_params = eval_config.label_with_params

    df_all = load_all_dfs(model_config.runs, res_path, checkpoints_to_expect, label_with_params)
    df_all.reset_index(drop=True, inplace=True)
    # print_aggragated_performance(df_all)
    # Plot the time to success from wandb
    df_wandb = pd.read_csv(wandb_log_file)
    df_wandb_dict = process_wandb_log(df_wandb, model_config, checkpoints_to_expect)
    os.makedirs(figure_dir, exist_ok=True)

    plot_time_to_success(df_all[df_all['dataset']=="test"], df_wandb_dict, percentages, checkpoints_to_expect, batch_size, args.colormap_path, figname=
    f"{figure_dir}/time_to_success_test_{figname_postfix}.pdf")
