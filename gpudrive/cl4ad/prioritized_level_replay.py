from typing import Any, List, Union
from collections import deque, defaultdict
from operator import itemgetter

import os 
import time
import pickle
import random
import numpy as np

from gpudrive.env.dataset import SceneDataLoader

from .base_curriculum import Curriculum

class PrioritizedLevelReplay(Curriculum):
    # Available score functions: 
    # amgae: Average magnitude of GAE. Check original PLR paper.
    # pvl: Positive value loss. Check PAIRED paper for Robust PLR.
    # maxmc: Maximum Monte Carlo. Check PAIRED paper for Robust PLR.
    # learn: Learnability-Hard. 
    # learn-goal: Learnability. Check Sampling for Learnability (SFL) paper.
    # gc-ade: Goal-conditioned average distance error
    # act-mae: Action mean absolute error
    SCORE_TYPES = [
        "amgae", "pvl", "maxmc", # Regret
        "learn", "learn-goal", # Learnability
        "gc-ade", "act-mae", # Realism
    ]
    MIN_RETURN = -1e6
    def __init__(
        self,
        data_loader: SceneDataLoader,
        gamma: float = 0.99,
        lambd: float = 0.95,
        buffer_size: int = 1000,
        replay_rate: float = 0.5,
        beta: float = 1.0,
        rho: float = 0.1,
        score_type: str = "amgae",
        warm_up_steps: int = 0,
        episode_length: int = 91,
        ):
        """
        Prioritized Level Replay samples scenes based on their priority.
        """
        super().__init__(data_loader)
        print(f"Using Prioritized Level Replay curriculum: score_type={score_type}, beta={beta}, rho={rho}, replay_rate={replay_rate}")
        self.gamma = gamma # Discount factor
        self.lambd = lambd # Lambda for GAE
        # Size of the replay buffer 
        self.buffer_size = buffer_size 
        if self.buffer_size > self.num_scenes:
            self.buffer_size = self.num_scenes
        self.replay_rate = replay_rate # Rate at which to replay scenes
        self.beta = beta # Temperature for ranking based distribution wrt scores
        self.rho = rho # Convex combination coefficient for staleness and scores

        self.score_type = score_type 
        # Validate the score type
        if self.score_type not in self.SCORE_TYPES:
            raise ValueError(
                f"Invalid score type: {self.score_type}. "
                "Must be one in SCORE_TYPES."
            )
        
        self.warm_up_steps = warm_up_steps # Number of warm-up steps before the curriculum starts

        self.episode_length = episode_length # Maximum length of an episode

        # Initialize the replay buffer
        self.buffer = np.array([],dtype=np.int32) # Buffer of scene ids
        self.scores = np.array([],dtype=np.float32) # Scores of scenes in the buffer
        self.staleness = np.array([],dtype=np.int32) # Staleness of scenes in the buffer
        self.seen_scenes = set() # Set of seen scenes: Useful for sampling unseen scenes
        self.scene_histogram = None

        # Initialize dictionary to store info until a rollout ends
        self.infos_dict = dict()
        self.num_agents = None
        self.scene_scores = dict()
        self.max_returns = None
        self.worlds_to_process = []

        # Initialize random generator
        self.seed = data_loader.seed
        self.random_gen = random.Random(self.seed)
        self.curriculum_iteration = 0
        self.previous_scenes = None
        self.current_scenes = None

        self.is_check_for_empty_scenes = False
        self.scenes_to_avoid = set()
        self.worlds_to_avoid = set()
    
    def __str__(self):
        return "plr"

    def save(self, path: str) -> None:
        """
        Save the buffer, scores, staleness, seen scenes, max returns, and curriculum iteration.
        :param path: Path to save the curriculum state.
        """
        state = {
            "buffer": self.buffer,
            "scores": self.scores,
            "staleness": self.staleness,
            "seen_scenes": self.seen_scenes,
            "max_returns": self.max_returns,
            "curriculum_iteration": self.curriculum_iteration,
            "previous_scenes": self.previous_scenes,
            "current_scenes": self.current_scenes,
            "scenes_to_avoid": self.scenes_to_avoid,
            "scene_histogram": self.scene_histogram,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Curriculum state saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the buffer, scores, staleness, seen scenes, max returns, and curriculum iteration
        :param path: Path to load the curriculum state from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Curriculum state file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.buffer = state["buffer"]
        self.scores = state["scores"]
        self.staleness = state["staleness"]
        self.seen_scenes = state["seen_scenes"]
        self.max_returns = state["max_returns"]
        self.curriculum_iteration = state["curriculum_iteration"]
        self.previous_scenes = state["previous_scenes"]
        self.current_scenes = state["current_scenes"]
        self.scenes_to_avoid = state["scenes_to_avoid"]
        self.scene_histogram = state["scene_histogram"]
        print(f"Curriculum state loaded from {path}")

    def _compute_values_and_ep_lens(self, rewards, values, dones, timesteps, max_returns):
        gammas = np.ones_like(rewards) * (self.gamma**np.arange(self.episode_length))
        dones_r = ~np.concatenate((
            np.zeros((dones.shape[0],1), dtype=dones.dtype), 
            dones[:,:-1]), axis=1)
        discounted_return = np.sum(rewards*gammas*dones_r, axis=1) # discounted returns
        initial_values = np.copy(discounted_return)
        continuing_episodes = np.sum(dones,axis=1) == 0
        greater_returns = discounted_return>max_returns
        if np.sum(continuing_episodes) == 0:
            # All episodes terminated
            ep_lens = np.cumsum(timesteps!=0,axis=1)[
                np.arange(rewards.shape[0]), np.sum(~dones,axis=1)]
            max_returns *= ~greater_returns 
            max_returns += discounted_return*greater_returns
            initial_values = np.copy(values[:,-1]) # Final value is the initial value as it follows reset()
        elif np.sum(continuing_episodes) == rewards.shape[0]:
            # No episodes terminated
            ep_lens = np.sum(timesteps!=0,axis=1)
            initial_values[continuing_episodes] += gammas[continuing_episodes,ep_lens[continuing_episodes]-1]*\
                values[continuing_episodes,ep_lens[continuing_episodes]-1]
        else:
            # Some episodes terminated
            ep_lens = np.zeros((timesteps.shape[0]), dtype=timesteps.dtype)
            ep_lens[continuing_episodes] = np.sum(timesteps!=0,axis=1)[continuing_episodes]
            ep_lens[~continuing_episodes] = timesteps[~continuing_episodes, 
                                                    np.argmax(dones[~continuing_episodes],axis=1)]
            max_returns[~continuing_episodes] = (max_returns*~greater_returns + discounted_return*greater_returns)[~continuing_episodes]
            initial_values[continuing_episodes] += gammas[continuing_episodes,ep_lens[continuing_episodes]]*\
                values[continuing_episodes,ep_lens[continuing_episodes]-1]
        return initial_values, ep_lens

    def _compute_gaes_for_scene(self, rewards, values, dones, timesteps, max_returns) -> dict:
        initial_values, ep_lens = self._compute_values_and_ep_lens(rewards, values, dones, timesteps, max_returns)
        vals = np.concatenate((initial_values[:,None], values), axis=1)
        deltas = np.zeros_like(rewards)
        ind_0 = np.repeat(np.arange(ep_lens.shape[0]), ep_lens)
        ind_1 = np.arange(ep_lens.sum()) - np.repeat(np.cumsum(ep_lens) - ep_lens, ep_lens)
        deltas[ind_0, ind_1] = rewards[ind_0, ind_1] + self.gamma*vals[ind_0, ind_1+1]*~dones[ind_0, ind_1] - vals[ind_0, ind_1]
        deltas[timesteps==0] *= 0.
        n, m = rewards.shape
        i = np.arange(m)[:, None]
        j = np.arange(m)[None, :]
        deltas_repeated = (
            np.broadcast_to(deltas[:, None, :], (n, m, m)) * \
                np.where(j >= i, (self.gamma*self.lambd) ** (j - i), 0.0)).reshape(n * m, m)  
        abs_sums = np.abs(deltas_repeated.sum(axis=1))
        max_sums = np.maximum(deltas_repeated.sum(axis=1), 0.0)
        starts = np.arange(0, abs_sums.size, deltas_repeated.shape[1])
        amgae = np.add.reduceat(abs_sums, starts) / m
        pvl = np.add.reduceat(max_sums, starts) / m
        maxmc_mask = np.zeros_like(vals)
        maxmc_mask[ind_0,ind_1] = 1.0
        maxmc = np.sum(maxmc_mask*(max_returns[:,None]-vals),axis=1)/ep_lens
        return {"amgae": amgae, "pvl": pvl, "maxmc": maxmc}, ep_lens

    def _compute_scores_for_scene(self, rewards, values, success, collision, off_road, timesteps, dones, d2lt, d2la, max_returns) -> dict:
        score_dict = {}
        gae_dict, ep_lens = self._compute_gaes_for_scene(rewards, values, dones, timesteps, max_returns)
        score_dict.update(gae_dict)
        # Learnability scores
        ind_0 = np.arange(ep_lens.shape[0])
        score_dict.update({
            "learn": 1.0 * success[ind_0, ep_lens-1] * (
                1.0 - (np.cumsum(collision, axis=1)[ind_0, ep_lens-1]>0)) * (
                    1.0 - (np.cumsum(off_road, axis=1)[ind_0, ep_lens-1]>0)),
            "learn-goal": 1.0 * success[ind_0, ep_lens-1],
            "gc-ade": np.sqrt(np.cumsum(d2lt, axis=1)[ind_0, ep_lens-1]) / ep_lens,
            "act-mae": np.cumsum(np.abs(d2la), axis=1)[ind_0, ep_lens-1] / ep_lens,
        })
        return score_dict
    
    def _compute_average_scores_for_scenes(self) -> None:
        """
        Compute the scores for all scenes in the current batch.
        :return: A tuple of (scene_ids_in_batch, scores_in_batch) where:
            scene_ids_in_batch: A numpy array of scene ids in the current batch.
            scores_in_batch: A numpy array of scores corresponding to the scene ids.
        """
        scene_ids_in_batch = np.array(list(self.scene_scores[self.score_type].keys()), dtype=np.int32)
        scores_in_batch = []
        for scene_id, scene_scores in self.scene_scores[self.score_type].items():    
            scores_in_batch.append(np.mean(scene_scores))
        self.scene_scores = dict() # Reset scene scores after computing them
        # Convert scores to numpy array
        scores_in_batch = np.array(scores_in_batch, dtype=np.float32)
        # Sort the scenes wrt their scores in descending order
        sorted_indices = np.argsort(scores_in_batch)[::-1]
        scene_ids_in_batch = scene_ids_in_batch[sorted_indices]
        scores_in_batch = scores_in_batch[sorted_indices]
        return scene_ids_in_batch, scores_in_batch

    def _update_buffer(self) -> None:
        """
        Update the replay buffer with the new scenes and their scores.
        :param scene_ids_in_batch: A numpy array of scene ids in the current batch.
        :param scores_in_batch: A numpy array of scores corresponding to the scene ids.
        """
        if len(self.scene_scores) == 0:
            return
        # Compute the scores of the scenes in the current batch
        scene_ids_in_batch, scores_in_batch = self._compute_average_scores_for_scenes()
        # If the buffer is empty, initialize it with the current batch
        scene_ids_in_batch_new, scores_in_batch_new = [], []
        for s_id, s_score in zip(scene_ids_in_batch, scores_in_batch):
            idx = np.where(self.buffer == s_id)[0]
            if idx.shape[0] > 0:
                self.scores[idx[0]] = s_score
                self.staleness[idx[0]] = self.curriculum_iteration
            else:
                scene_ids_in_batch_new.append(s_id)
                scores_in_batch_new.append(s_score)
        self.buffer = np.concatenate((self.buffer, np.array(scene_ids_in_batch_new, dtype=np.int32)))
        self.scores = np.concatenate((self.scores, np.array(scores_in_batch_new, dtype=np.float32)))
        self.staleness = np.concatenate((self.staleness,
                                            [self.curriculum_iteration] * len(scene_ids_in_batch_new)))
        # Sort the buffer with respect to the scores in descending order
        sorted_indices = np.argsort(self.scores)[::-1]
        self.buffer = self.buffer[sorted_indices[:self.buffer_size]]
        self.scores = self.scores[sorted_indices[:self.buffer_size]]
        self.staleness = self.staleness[sorted_indices[:self.buffer_size]]
        
    def _update_infos_dict(self, infos) -> None:
        """
        Update the infos_dict with the new information from the current step.
        This method is called after each step in the environment to store information
        necessary to compute scene scores.
        :param infos: A dictionary containing information about the current step.
        """

        self.worlds_to_process = [dw for dw in infos["done_worlds"].tolist() if dw not in self.worlds_to_avoid]
        if self.num_agents is None:
            self.num_agents = np.bincount(infos["worlds"])
            if self.num_agents.shape[0] < self.batch_size:
                self.num_agents = np.concatenate((self.num_agents, 
                                                  np.zeros(self.batch_size-self.num_agents.shape[0], 
                                                           dtype=self.num_agents.dtype)))
            self.max_returns = np.ones(self.num_agents.sum())*self.MIN_RETURN
        for info_key, info_val in infos.items():
            if info_key == "done_worlds":
                continue
            if info_key not in self.infos_dict:
                # info_val.shape[0] is the number of all agents across all worlds
                self.infos_dict[info_key] = np.zeros(
                    (info_val.shape[0]*self.episode_length,), dtype=info_val.dtype
                )
            self.infos_dict[info_key][
                np.arange(info_val.shape[0], 
                          dtype=np.int32) * \
                            self.episode_length + infos['timesteps'] - 1] = info_val

    def _process_worlds(self) -> None:
        """
        Process the worlds that have completed their episodes or are about to be flushed.
        This method computes the scores for each scene based on the information
        stored in infos_dict and updates the scene_scores dictionary.
        """
        if len(self.worlds_to_process) == 0:
            return
        for w in self.worlds_to_process:
            if w in self.worlds_to_avoid:
                continue
            # scene_scores = defaultdict(list)
            w_idx = np.zeros(self.num_agents.sum()*self.episode_length, dtype=bool)
            w_start_idx = np.sum(self.num_agents[:w])*self.episode_length
            scene = self.infos_dict['scenes'][w_start_idx]
            w_idx[w_start_idx:w_start_idx+self.num_agents[w]*self.episode_length] = True
            scene_scores = self._compute_scores_for_scene(
                rewards=self.infos_dict['rewards'][w_idx].reshape(-1,self.episode_length),
                values=self.infos_dict['values'][w_idx].reshape(-1,self.episode_length),
                success=self.infos_dict['success'][w_idx].reshape(-1,self.episode_length),
                collision=self.infos_dict['collision'][w_idx].reshape(-1,self.episode_length),
                off_road=self.infos_dict['off-road'][w_idx].reshape(-1,self.episode_length),
                timesteps=self.infos_dict['timesteps'][w_idx].reshape(-1,self.episode_length),
                dones=self.infos_dict['dones'][w_idx].reshape(-1,self.episode_length),
                d2lt=self.infos_dict['d2lt'][w_idx].reshape(-1,self.episode_length),
                d2la=self.infos_dict['d2la'][w_idx].reshape(-1,self.episode_length),
                max_returns=self.max_returns[self.num_agents[:w].sum():self.num_agents[:w+1].sum()],
            )
            # Compute the average score of the agent across all episodes, and add it to the scene's score.
            for score_type, scores in scene_scores.items():
                self.scene_scores.setdefault(score_type, {}).setdefault(scene, [])
                if "learn" in score_type:
                    # For learnability, we do not average the score by the number of agents.
                    self.scene_scores[score_type][scene].append(
                        np.mean(scores) * (1 - np.mean(scores))
                    )
                else:
                    # For other scores, we average the score by the number of agents.
                    self.scene_scores[score_type][scene].append(np.mean(scores))   
            # Rest the processed world in infos_dict
            for info_key in self.infos_dict:
                self.infos_dict[info_key][w_idx] *= np.zeros((1,), dtype=self.infos_dict[info_key].dtype)

    def _check_for_empty_scenes(self, infos_scenes) -> None:
        if not self.is_check_for_empty_scenes:
            current_unique_scenes_ids = set([
                self.file_to_index[cus] for cus in set(self.current_scenes)
                ]) - self.scenes_to_avoid
            for cus_id in current_unique_scenes_ids:
                if np.where(infos_scenes == cus_id)[0].shape[0] == 0:
                    print(f"Scene without controllable object: {self.data_loader.dataset[cus_id]}")
                    self.scenes_to_avoid.add(cus_id)
            for w in range(self.batch_size):
                if self.file_to_index[self.current_scenes[w]] in self.scenes_to_avoid:
                    self.worlds_to_avoid.add(w)
            self.is_check_for_empty_scenes = True

    def update(self, infos) -> None:
        """
        Update the curriculum with new information.
        This method is called after each step in the environment to store information
        necessary to compute scene scores.
        :param infos: A dictionary containing information about the current step with keys:
            'timesteps', 'scenes', 'agents', 'worlds',  'dones', 
            'rewards', 'values', 'success', 'collision', 'off-road',
            'd2lt', 'd2la'
        """
        self._check_for_empty_scenes(infos["scenes"])
        self._update_infos_dict(infos)
        self._process_worlds()

    def _uniform_distribution_over_unseen_scenes(self) -> np.ndarray:
        """Returns a uniform distribution over unseen scenes.
        This distribution is used when the buffer is empty or when there are unseen scenes
        and the decision is to replay from unseen scenes.
        :return: A numpy array of shape (num_scenes,) representing the uniform distribution over unseen scenes.
        """
        min_scene_id = np.min(self.unique_scenes)
        unseen_scenes = np.array(list(set(self.unique_scenes) - self.seen_scenes - self.scenes_to_avoid))
        uniform_distribution = np.zeros(len(self.unique_scenes), dtype=np.float32)
        uniform_distribution[unseen_scenes-min_scene_id] = 1.0 / unseen_scenes.shape[0]
        return uniform_distribution
    
    def _plr_distribution(self) -> np.ndarray:
        """Returns the prioritized level replay distribution over the buffer.
        This distribution is a convex combination of the score-based distribution and the staleness-based distribution.
        :return: A numpy array of shape (num_scenes,) representing the PLR distribution over the buffer.
        """
        # Buffer is already ordered (descending) with respect to scores
        min_scene_id = np.min(self.unique_scenes)
        # Score distribution is computed based on the ranking of the scores. See PLR paper for P_S.
        score_distribution = np.zeros(len(self.unique_scenes), dtype=np.float32)
        score_distribution[self.buffer - min_scene_id] = (1 / np.arange(1, self.buffer.shape[0]+1))**(1/self.beta)
        # Staleness distribution is computed based on the staleness of the scenes in the buffer. See PLR paper for P_C.
        staleness_distribution = np.zeros(len(self.unique_scenes), dtype=np.float32)
        staleness_distribution[self.buffer - min_scene_id] = self.curriculum_iteration - self.staleness + 1
        for scene_to_avoid in self.scenes_to_avoid:
            score_distribution[scene_to_avoid - min_scene_id] = 0.0
            staleness_distribution[scene_to_avoid - min_scene_id] = 0.0
        score_distribution /= np.sum(score_distribution)
        staleness_distribution /= np.sum(staleness_distribution)
        # Convex combination of both distributions is the replay distribution
        replay_distribution = (1 - self.rho) * score_distribution + self.rho * staleness_distribution
        return replay_distribution

    def _sample_distribution(self) -> np.ndarray:
        """Returns a sample distribution over the task space.

        Any curriculum that maintains a true probability distribution should implement this method to retrieve it.
        """
        # If the buffer is empty or if the decision is not to replay when there are unseen scenes
        # then uniformly sample over unseen scenes
        if self.buffer.shape[0] == 0 or (
            len(set(self.unique_scenes) - self.seen_scenes) > 0 and \
                self.random_gen.random() < self.replay_rate):
            return self._uniform_distribution_over_unseen_scenes()
        else:
            # Sample from the buffer based on the scores and staleness
            return self._plr_distribution()
    
    def _process_incomplete_episodes(self) -> None:
        """Process any incomplete episodes in the infos_dict.
        This method processes any worlds that have not yet been processed but have
        incomplete episodes. It ensures that all worlds are processed before sampling.
        """
        # Early exit if there is nothing to process or if all worlds have been processed already
        if len(self.infos_dict) == 0 or len(self.worlds_to_process) == self.batch_size:
            return
        # Process any worlds that have not yet been processed
        self.worlds_to_process = list(set(list(range(self.batch_size))) - \
                                      set(self.worlds_to_process) - \
                                        self.worlds_to_avoid)
        self._process_worlds()

    def _update_scene_histogram(self, current_scenes_ids=None):
        if current_scenes_ids is None:
            current_scene_histogram = self.scene_histogram[-1:,:]
        else:
            min_scene_id = np.min(self.unique_scenes)
            current_scenes_counts = np.bincount(np.array(current_scenes_ids) - min_scene_id)
            max_current_scenes_id = np.max(current_scenes_ids)
            current_scene_histogram = np.zeros((1,len(self.unique_scenes)), dtype=np.int32)
            current_scene_histogram[-1,:max_current_scenes_id-min_scene_id+1] += current_scenes_counts
        if self.scene_histogram is None:
            self.scene_histogram = current_scene_histogram
        else:
            self.scene_histogram = np.concatenate((self.scene_histogram, current_scene_histogram),axis=0)
            
    def sample(self) -> Union[List, Any]:
        """Sample scenes based on the prioritized level replay distribution.
        :return: A list of scene ids sampled from the curriculum.
        """
        self.previous_scenes = getattr(self, 'current_scenes', None)
        self._process_incomplete_episodes()
        self.infos_dict = dict()
        self.num_agents = None
        # Update the buffer with the new scenes and their scores
        self._update_buffer()
        # Sample at the beginning of the curriculum or after warm-up steps
        if self.curriculum_iteration == 0 or self.curriculum_iteration >= self.warm_up_steps:
            print(f"Sampling scenes from the curriculum at step {self.curriculum_iteration}...")
            # Sample a distribution over the task space
            distribution = self._sample_distribution()
            # Sample scenes based on the distribution
            current_scenes_ids = self.random_gen.choices(self.unique_scenes, 
                                                         weights=distribution.tolist(), 
                                                         k=self.batch_size)
            self.current_scenes = [self.data_loader.dataset[s_id] for s_id in current_scenes_ids]
            self._update_scene_histogram(current_scenes_ids)
        else:
            print(f"Using previously sampled scenes at step {self.curriculum_iteration}...")
            self._update_scene_histogram()
        self.is_check_for_empty_scenes = False
        self.worlds_to_avoid = set()
        self.seen_scenes.update(set(
            [self.file_to_index[s] for s in self.current_scenes]
        ))
        self.curriculum_iteration += 1
        self.random_gen = random.Random(self.seed + self.curriculum_iteration)
        return self.current_scenes