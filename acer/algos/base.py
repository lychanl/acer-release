from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import numpy as np

from environment import BaseMultiEnv


class Agent(ABC):
    """RL algorithm abstraction"""

    @abstractmethod
    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, float, np.array, np.array, bool, bool]
    ]):
        ...

    @abstractmethod
    def learn(self):
        ...

    @abstractmethod
    def predict_action(self, observations: np.array, is_deterministic: bool = False) -> Tuple[list, np.array]:
        ...


