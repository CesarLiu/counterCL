import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import torch
from tqdm import tqdm
from gpudrive.env.dataset import SceneDataLoader
from eval_utils import (
    load_config,
    make_env,
)
from gpudrive.visualize.utils import img_from_fig

def get_controlled_agent_counts(eval_config, fname=None):
    if fname is not None and os.path.exists(fname):
        with open(fname, "r") as f:
            scenes_txt = f.readlines()
        scenes = [scene.split(",")[0] for scene in scenes_txt]
        num_controlled_agents = [int(scene.split(",")[1]) for scene in scenes_txt]
        return scenes, num_controlled_agents
    else:
        env = make_env(eval_config, 
                    SceneDataLoader(
                        root=eval_config.train_dir,
                        batch_size=eval_config.num_worlds,
                        dataset_size=eval_config.num_worlds,
                        sample_with_replacement=False,))
        data_loader = SceneDataLoader(
            root=eval_config.train_dir,
            batch_size=eval_config.num_worlds,
            dataset_size=eval_config.train_dataset_size,
            sample_with_replacement=False,
            shuffle=False,
        )

        num_controlled_agents = []
        scenes = []
        for batch in tqdm(
            data_loader,
            desc=f"Processing training batches",
            total=len(data_loader),
            colour="blue",
        ):
            env.swap_data_batch(batch)
            env.reset()
            num_controlled_agents.extend(torch.sum(env.cont_agent_mask, dim=1).cpu().numpy().tolist())
            scenes.extend(batch)
        
        with open(fname, "w") as f:
            for scene, n in zip(scenes, num_controlled_agents):
                f.write(f"{scene},{n}\n")
        return scenes, num_controlled_agents

def get_initial_frames(scenarios, eval_config):
    eval_config.num_worlds = len(scenarios)
    env = make_env(eval_config, 
                   SceneDataLoader(
                       root=eval_config.train_dir,
                       batch_size=len(scenarios),
                       dataset_size=eval_config.train_dataset_size,
                       sample_with_replacement=False,),
                       render_3d=True,
                       )

    env.swap_data_batch(scenarios)
    env.reset()
    env.step_dynamics(env.get_expert_actions()[0][:, :, 0, :])
    sim_state_figures = env.vis.plot_simulator_state(
        env_indices=list(range(len(scenarios))),
        time_steps=[0] * len(scenarios),
        zoom_radius=30,
        center_agent_indices=torch.argmax(env.cont_agent_mask.to(torch.uint8), dim=1).tolist(),
    )
    sim_state_frames = {scenario: [] for scenario in scenarios}
    initial_frame_count = 0
    for idx, scenario in enumerate(scenarios):
        sim_state_frames[scenario].append(
            np.array(img_from_fig(sim_state_figures[idx]))
        )
        initial_frame_count += 1
        if initial_frame_count >= 3:
            break
    return sim_state_frames

def compute_replay_distribution(curriculum, dataset_size, beta, rho):
    # Buffer is already ordered (descending) with respect to scores
    scene_to_avoid = curriculum["scenes_to_avoid"]
    buffer = curriculum["buffer"]
    scores = curriculum["scores"]
    staleness = curriculum["staleness"]
    curriculum_iteration = curriculum["curriculum_iteration"]
    # Score distribution is computed based on the ranking of the scores. See PLR paper for P_S.
    score_distribution = np.zeros(dataset_size, dtype=np.float32)
    score_distribution[buffer] = (1 / np.arange(1, buffer.shape[0]+1))**(1/beta)
    # Staleness distribution is computed based on the staleness of the scenes in the buffer. See PLR paper for P_C.
    staleness_distribution = np.zeros(dataset_size, dtype=np.float32)
    staleness_distribution[buffer] = curriculum_iteration - staleness + 1
    for scene_to_avoid in scene_to_avoid:
        score_distribution[scene_to_avoid] = 0.0
        staleness_distribution[scene_to_avoid] = 0.0
    score_distribution /= np.sum(score_distribution)
    staleness_distribution /= np.sum(staleness_distribution)
    # Convex combination of both distributions is the replay distribution
    replay_distribution = (1 - rho) * score_distribution + rho * staleness_distribution
    return replay_distribution

if __name__ == "__main__":
    eval_config = load_config("examples/experimental/config/eval_config_1000scenes")
    model_config = load_config("examples/experimental/config/model_config_plr_pvl_1000scenes")
    scene_txt_path = "mini_train_scenes.txt"
    figname_postfix = "_pvl"
    checkpoints = [200 + i*400 for i in range(10)]

    # # Read the scene txt file
    # scenes = []
    # with open(scene_txt_path, "r") as f:
    #     scenes_txt = f.readlines()
    # scenes = [scene.split(",")[0] for scene in scenes_txt]
    # num_controlled_agents = [int(scene.split(",")[1]) for scene in scenes_txt]
    scenes, num_controlled_agents = get_controlled_agent_counts(eval_config, fname=scene_txt_path)

    file_to_index = {
            file: idx for idx, file in enumerate(scenes)
    }

    num_controlled_agents = np.array(num_controlled_agents)
    runs_dict = {}
    for run in model_config.runs:
        runs_dict[run.name] = {"beta": run.beta, "rho": run.rho, "seed": run.train_seed, "curriculum": {}}
        curriculum_path = os.path.join("runs", run.name)
        for f in os.listdir(curriculum_path):
            ckpt = int(f[-9:-4])
            if f.endswith(".pkl") and ckpt in checkpoints:
                curriculum_file = os.path.join(curriculum_path, f)
                with open(curriculum_file, "rb") as f:
                    curriculum = pickle.load(f)
                runs_dict[run.name]["curriculum"][ckpt] = curriculum

    # Compute replay distribution for each run at each checkpoint
    replay_distributions = {}
    for run_name, run_dict in tqdm(runs_dict.items(), desc="Computing replay distributions"):
        replay_distributions[run_name] = []
        for ckpt, curriculum in run_dict["curriculum"].items():
            replay_distributions[run_name].append(compute_replay_distribution(curriculum, len(scenes), run_dict["beta"], run_dict["rho"]))
        replay_distributions[run_name] = np.array(replay_distributions[run_name])

    # Find top 2 scenes with max probability at each checkpoint
    # Then print the scenes that appear in all runs at any checkpoint
    top_3_scenes = np.zeros((len(replay_distributions), len(checkpoints), 3))
    for i, (run_name, replay_distribution) in enumerate(replay_distributions.items()):
        for ckpt_idx in range(replay_distribution.shape[0]):
            top_3_scenes[i, ckpt_idx, :] = np.argsort(replay_distribution[ckpt_idx,:])[-3:]
    print(top_3_scenes)
    unique_scenes = np.unique(top_3_scenes[0,:,:].reshape(-1))
    scenes_that_occur_in_all_runs = []
    # occurence_ckpt_indices = np.zeros((len(replay_distributions), 3))
    for scene in unique_scenes:
        number_of_occurrences = 0
        for i, run_name in enumerate(replay_distributions.keys()):
            number_of_occurrences += np.sum(top_3_scenes[i,:,:] == scene) > 0
        if number_of_occurrences == len(replay_distributions):
            scenes_that_occur_in_all_runs.append(int(scene))
            print(f"Scenario {scene} with {num_controlled_agents[int(scene)]}-many cars occurs in all runs at any checkpoint")

    initial_frames = get_initial_frames([scenes[sc] for sc in scenes_that_occur_in_all_runs[:3]], eval_config)

    # Keep rows close to each other
    # matplotlib.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(3, len(replay_distributions), figsize=(10, 10))
    for i, (run_name, replay_distribution) in enumerate(replay_distributions.items()):
        c = axes[0,i].imshow(replay_distribution.T, aspect='auto', origin='lower', cmap='Reds')
        if i == len(replay_distributions) - 1:
            cbar = plt.colorbar(c, ax=axes[0,i], label="Replay Probability")
        else:
            cbar = plt.colorbar(c, ax=axes[0,i])
        # Configure the colorbar formatter for scientific notation
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()
        axes[0,i].set_xticks(np.arange(replay_distribution.shape[0]))
        axes[0,i].set_xticklabels(checkpoints, rotation=45)
        for x in range(replay_distribution.shape[0] + 1):
            axes[0,i].axvline(x - 0.5, color="w", linewidth=1.5) 
        if i != 0:
            axes[0,i].set_yticks([])
        axes[0,i].set_xlabel("Number of policy updates")
        axes[0,i].set_title(f"seed: {runs_dict[run_name]['seed']}")
    for i, (run_name, replay_distribution) in enumerate(replay_distributions.items()):
        num_controlled_agents_distribution = np.zeros((max(num_controlled_agents) + 1, replay_distribution.shape[0]))
        for n in range(max(num_controlled_agents) + 1):
            scenes_with_n_controlled_agents = np.array(num_controlled_agents) == n
            num_controlled_agents_distribution[n,:] = np.sum(replay_distribution[:,scenes_with_n_controlled_agents], axis=1)
        c = axes[1,i].imshow(num_controlled_agents_distribution, aspect='auto', origin='lower', cmap='Reds')
        if i == len(replay_distributions) - 1:
            cbar = plt.colorbar(c, ax=axes[1,i], label="Replay Probability")
        else:
            cbar = plt.colorbar(c, ax=axes[1,i])
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.update_ticks()
        axes[1,i].set_xticks(np.arange(num_controlled_agents_distribution.shape[1]))
        axes[1,i].set_xticklabels(checkpoints)
        for x in range(num_controlled_agents_distribution.shape[1] + 1):
            axes[1,i].axvline(x - 0.5, color="w", linewidth=1.5)
        if i != 0:
            axes[1,i].set_yticks([])
        # Labels
        axes[1,i].set_xlabel("Number of policy updates")
        axes[1,i].set_xticks(np.arange(num_controlled_agents_distribution.shape[1]))
        axes[1,i].set_xticklabels(checkpoints, rotation=45)
        axes[1,i].set_title(f"seed: {runs_dict[run_name]['seed']}")
    axes[1,0].set_ylabel("Number of controlled agents")
        # axes[1,i].set_title(run_name)
    for i, (scenario, initial_frame) in enumerate(initial_frames.items()):
        axes[2,i].imshow(np.rot90(initial_frame[0], k=2), aspect='auto', origin='lower')
        axes[2,i].set_title(f"Scenario {scenes.index(scenario)}")
        axes[2,i].set_yticks([])
        axes[2,i].set_xticks([])
        axes[2,i].set_xlabel(f"Number of controlled agents: {num_controlled_agents[scenes.index(scenario)]}")
    axes[0,0].set_ylabel("Scenario IDs")
    plt.tight_layout(h_pad=0.5)
    plt.savefig(f"{eval_config.figure_dir}/replay_distributions_{eval_config.figname_postfix}{figname_postfix}.pdf", dpi=500, bbox_inches='tight', pad_inches=0.1)
    plt.close()