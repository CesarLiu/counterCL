import os
import random
import pickle 
from .base_curriculum import Curriculum
from gpudrive.env.dataset import SceneDataLoader

class DefaultCurriculum(Curriculum):
    def __init__(self, data_loader: SceneDataLoader):
        """
        A default curriculum samples scenes by iterating the data_loader
        :param data_loader: SceneDataLoader instance that provides the scenes to sample from.
        """
        super().__init__(data_loader)
        self.data_iterator = iter(self.data_loader)
        self.sample_no = 0
        self.previous_scenes = None
        self.current_scenes = None

    def __str__(self):
        return "default"
    
    def save(self, path: str) -> None:
        # Save sample number, previous and current scenes, current index of data iterator
        state = {
            "sample_no": self.sample_no,
            "previous_scenes": self.previous_scenes,
            "current_scenes": self.current_scenes,
            "current_index": self.data_iterator.current_index,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Curriculum state saved to {path}")

    def load(self, path: str) -> None:
        # Load sample number, previous and current scenes, current index of data iterator
        if not os.path.exists(path):
            raise FileNotFoundError(f"Curriculum state file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.sample_no = state["sample_no"]
        self.previous_scenes = state["previous_scenes"]
        self.current_scenes = state["current_scenes"]
        self._reset_data_iterator()
        self.data_iterator.current_index = state["current_index"]
        print(f"Curriculum state loaded from {path}")

    def update(self, infos) -> None:
        pass

    def _reset_data_iterator(self):
        # Reset random generator to ensure reproducibility
        self.data_loader.random_gen = random.Random(self.data_loader.seed)
        # Reset the data iterator to iterate through same scenes again
        self.data_iterator = iter(self.data_loader)

    def sample(self):
        """
        Returns the next item from the iterator. If the iterator is exhausted, it resets.

        :return: The next item from the data_loader.
        """
        self.previous_scenes = self.current_scenes
        if self.sample_no == 0:
            self._reset_data_iterator()
        try:
            self.current_scenes = next(self.data_iterator)
        except StopIteration:
            # Reset the iterator if it is exhausted
            self._reset_data_iterator()
            self.current_scenes = next(self.data_iterator)

        self.sample_no += 1
        return self.current_scenes