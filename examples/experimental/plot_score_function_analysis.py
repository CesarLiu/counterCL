import os
import numpy as np
import pandas as pd
import seaborn as sns
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
        print(f"\nProcessing file: {df_filename}")
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
        print(f"method: {method}")
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

def direction_to_symbol(direction):
    if direction == "up":
        return r"$\uparrow$"
    elif direction == "down":
        return r"$\downarrow$"
    else:
        return ""

def order_methods(agg):
    return sorted(agg['method'].unique(), reverse=True)

def get_aggragated_df(df, metrics_to_plot_list):
    df_long = df.melt(
        id_vars=['method', 'checkpoint', 'scene', 'train_seed'],
        value_vars=metrics_to_plot_list,
        var_name='metric',
        value_name='value'
    )
    # Step 1: average over seeds per scene
    per_scene = df_long.groupby(
        ['method', 'checkpoint', 'train_seed', 'metric']
    )['value'].mean().reset_index()

    # Step 2: compute mean and std over scenes
    agg = per_scene.groupby(
        ['method', 'checkpoint', 'metric']
    )['value'].agg(['mean', 'std']).reset_index()

    return agg

def plot_shadow_plot(df, metrics_to_plot, ordered_methods, colormap_path, figname):
    metrics_to_plot_names = [name for _, name, _ in metrics_to_plot]
    metrics_to_plot_list = [metric for metric, _, _ in metrics_to_plot]
    metrics_to_plot_directions = [direction for _, _, direction in metrics_to_plot]

    agg = get_aggragated_df(df, metrics_to_plot_list)
    checkpoints = sorted(agg['checkpoint'].unique())
    methods = sorted(agg['method'].unique())
    n_methods = len(methods)
    palette = sns.color_palette(n_colors=n_methods)
    if os.path.exists(colormap_path):
        import json
        with open(colormap_path, 'r') as f:
            color_map = json.load(f)
        print("Using provided color map: ", color_map)
    else:
        color_map = {m: palette[i] for i, m in enumerate(methods)}
    markers = ['o', 's', 'D', 'v', '^', 'P', 'H', 'X', 'd', 'p']
    marker_map = {m: markers[i] for i, m in enumerate(methods)}
    matplotlib.rcParams.update({'font.size': 16})
    g = sns.FacetGrid(agg, col='metric', sharey=False, height=4, aspect=1.2)
    axes = g.axes.flat
    for ax_idx, (metric, subdf) in enumerate(agg.groupby('metric')):
        ax = axes[metrics_to_plot_list.index(metric)]
        for i, method in enumerate(ordered_methods):
            mdf = subdf[subdf['method'] == method]
            ax.plot(mdf['checkpoint'], mdf['mean'], label=method_to_label(method), marker=marker_map[method], linestyle='-', color=color_map[method])
            ax.fill_between(mdf['checkpoint'], mdf['mean'] - mdf['std'], mdf['mean'] + mdf['std'], alpha=0.2, color=color_map[method])

        ax.set_title(metrics_to_plot_names[metrics_to_plot_list.index(metric)] + f" {direction_to_symbol(metrics_to_plot_directions[metrics_to_plot_list.index(metric)])}")
        every_k_ckpts = 1 if len(checkpoints) <= 10 else 2
        ax.set_xticks([ckpt for ckpt_i, ckpt in enumerate(checkpoints) if ckpt_i % every_k_ckpts == 0])
        ax.tick_params(axis='y', which='both', labelleft=True)
        ax.tick_params(axis='x', which='both', rotation=30)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax.set_xlabel("Number of policy updates")

    # Remove default legend (if any)
    if getattr(g, "_legend", None) is not None:
        g._legend.remove()

    # Build manual legend handles (guaranteed labels)
    handles = [plt.Line2D([0], [0], color=color_map[m], marker=marker_map[m], linestyle='-', label=method_to_label(m)) for m in ordered_methods]
    g.fig.legend(
        handles=handles,
        labels=[method_to_label(m) for m in ordered_methods],
        title="Method",
        loc='lower center',
        ncol=min(len(methods), 2*len(metrics_to_plot)), # Every two metrics deletes a column from 4 columns
        bbox_to_anchor=(0.5, 0.85)
    )

    # Adjust layout so legend has room
    g.fig.subplots_adjust(top=0.84)
    g.fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Save without clipping
    plt.savefig(figname, dpi=500, bbox_inches='tight', pad_inches=0.1)
    plt.close(g.fig)


if __name__ == "__main__":
    # Load configurations
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/experimental/config/eval_config_default_1000scenes_computeab")
    parser.add_argument("--model_config", type=str, default="examples/experimental/config/model_config_default_1000scenes_computeab")
    parser.add_argument("--colormap_path", type=str, default="color_map.json")
    parser.add_argument("--res_path", type=str)
    parser.add_argument("--figure_dir", type=str)
    parser.add_argument("--figname_postfix", type=str)
    args = parser.parse_args()

    eval_config = load_config(args.config)
    model_config = load_config(args.model_config)

    passed_args = {
        "res_path": args.res_path,
        "figure_dir": args.figure_dir,
        "figname_postfix": args.figname_postfix,
    }
    eval_config.update(
        {k: v for k, v in passed_args.items() if v is not None}
    )

    res_path = eval_config.res_path
    figure_dir = eval_config.figure_dir
    figname_postfix = eval_config.figname_postfix
    checkpoints_to_expect = [eval_config.first_ckpt+i*eval_config.every_k_ckpts for i in range(eval_config.number_of_ckpts)] 
    label_with_params = eval_config.label_with_params

    df_all = load_all_dfs(model_config.runs, res_path, checkpoints_to_expect, label_with_params)
    df_all.reset_index(drop=True, inplace=True)
    # print_aggragated_performance(df_all)
    os.makedirs(figure_dir, exist_ok=True)

    metrics_to_plot = [
        ("disc_return", "Discounted Return", "up"),
        ("goal_achieved_frac", "Success Rate", "up"),
        ("collided_frac", "Collision Rate", "down"),
        ("off_road_frac", "Off-Road Rate", "down"),
    ]
    ordered_methods = order_methods(get_aggragated_df(df_all[df_all['dataset']=="test"], ["goal_achieved_frac"]))
    print(ordered_methods)
    if eval_config.eval_train_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="train"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/performance_shadow_plots_train_{figname_postfix}.pdf")
    if eval_config.eval_test_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="test"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/performance_shadow_plots_test_{figname_postfix}.pdf")
    if eval_config.eval_valid_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="valid"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/performance_shadow_plots_valid_{figname_postfix}.pdf")

    metrics_to_plot = [
        ("amgae", "Average Magnitude of GAE", ""),
        ("pvl", "Positive Value Loss", ""),
        ("maxmc", "Maximum Monte Carlo", ""),
    ]
    if eval_config.eval_train_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="train"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/regret_shadow_plots_train_{figname_postfix}.pdf")
    if eval_config.eval_test_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="test"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/regret_shadow_plots_test_{figname_postfix}.pdf")
    if eval_config.eval_valid_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="valid"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/regret_shadow_plots_valid_{figname_postfix}.pdf")

    metrics_to_plot = [
        ("learn", "Learnability-Hard", ""),
        ("learn_goal", "Learnability", ""),
    ]
    if eval_config.eval_train_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="train"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/learnability_shadow_plots_train_{figname_postfix}.pdf")
    if eval_config.eval_test_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="test"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/learnability_shadow_plots_test_{figname_postfix}.pdf")
    if eval_config.eval_valid_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="valid"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/learnability_shadow_plots_valid_{figname_postfix}.pdf")

    metrics_to_plot = [
        ("gc_ade", "Goal-Cond. Ave. Distance Err.", "down"),
        # ("act_mae", "Action Mean Absolute Error", "down"),
    ]
    if eval_config.eval_train_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="train"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/realism_shadow_plots_train_{figname_postfix}.pdf")
    if eval_config.eval_test_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="test"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/realism_shadow_plots_test_{figname_postfix}.pdf")
    if eval_config.eval_valid_dataset:
        plot_shadow_plot(df_all[df_all['dataset']=="valid"], metrics_to_plot, ordered_methods, args.colormap_path, f"{figure_dir}/realism_shadow_plots_valid_{figname_postfix}.pdf")