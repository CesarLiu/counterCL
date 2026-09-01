from typing import Any, List, Union
from gpudrive.env.dataset import SceneDataLoader

class Curriculum:
    def __init__(
        self,
        data_loader: SceneDataLoader,
    ):
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        # self.unique_scenes = list(set(data_loader.dataset))
        self.file_to_index = {
            file: idx for idx, file in enumerate(self.data_loader.dataset)
        }
        self.unique_scenes = list(self.file_to_index.values())
        self.unique_scenes.sort()  # Ensure consistent ordering of scenes

    @property
    def num_scenes(self) -> int:
        """Number of unique scenes in the curriculum."""
        return len(self.unique_scenes)
    
    def __str__(self) -> str:
        """String representation of the curriculum.

        Returns:
            str: A string describing the curriculum.
        """
        raise NotImplementedError(
            "Curriculum __str__ method must be implemented in subclasses."
        )
    
    
    def save(self, path: str) -> None:
        """Saves the curriculum state to a file.

        Args:
            path (str): Path to save the curriculum state.
        """
        raise NotImplementedError(
            "Curriculum save method must be implemented in subclasses."
        )
    
    def load(self, path: str) -> None:
        """Loads the curriculum state from a file.

        Args:
            path (str): Path to load the curriculum state from.
        """
        raise NotImplementedError(
            "Curriculum load method must be implemented in subclasses."
        )

    def update(self, infos) -> None:
        raise NotImplementedError(
            "Curriculum update method must be implemented in subclasses."
        )
    
    def _sample_distribution(self) -> List[float]:
        """Returns a sample distribution over the task space.

        Any curriculum that maintains a true probability distribution should implement this method to retrieve it.
        """
        raise NotImplementedError
    
    def sample(self) -> Union[List, Any]:
        """Samples a batch of scenes from the curriculum.
        This method should be implemented in subclasses to provide specific sampling strategies.
        """
        raise NotImplementedError(
            "Curriculum sample method must be implemented in subclasses."
        )

        