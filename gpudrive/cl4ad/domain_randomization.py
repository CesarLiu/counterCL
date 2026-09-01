import os
import pickle
import random
import numpy as np

from .base_curriculum import Curriculum
from gpudrive.env.dataset import SceneDataLoader

class DomainRandomization(Curriculum):
    def __init__(self, data_loader: SceneDataLoader):
        """
        Domain Randomization samples scenes uniformly randomly

        :param data_loader: SceneDataLoader instance that provides the scenes to sample from.
        """
        super().__init__(data_loader)
        self.seed = data_loader.seed
        self.random_gen = random.Random(self.seed)
        self.sample_no = 0
        self.previous_scenes = None
        self.current_scenes = None

    def __str__(self):
        return "domain_randomization"
    
    def save(self, path: str) -> None:
        # Save sample number, previous and current scenes
        state = {
            "sample_no": self.sample_no,
            "previous_scenes": self.previous_scenes,
            "current_scenes": self.current_scenes,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Curriculum state saved to {path}")

    def load(self, path: str) -> None:
        # Load sample number, previous and current scenes
        if not os.path.exists(path):
            raise FileNotFoundError(f"Curriculum state file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.sample_no = state["sample_no"]
        self.previous_scenes = state["previous_scenes"]
        self.current_scenes = state["current_scenes"]
        # Reset random generator to ensure reproducibility
        self.random_gen = random.Random(self.seed + self.sample_no)
        print(f"Curriculum state loaded from {path}")

    def update(self, infos) -> None:
        pass

    def sample(self):
        """
        Samples a batch of scenes uniformly at random from the unique scenes.
        :return: A list of randomly sampled scenes.
        """
        self.previous_scenes = self.current_scenes
        self.current_scenes = [
            self.data_loader.dataset[
            self.unique_scenes[
                self.random_gen.randint(0, self.num_scenes - 1)
            ]
            ]
            for _ in range(self.batch_size)
        ]
        self.sample_no += 1
        self.random_gen.seed(self.seed + self.sample_no)

        return self.current_scenes