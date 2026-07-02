from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    """Abstract interface for Franka arm controllers.

    Subclass this to implement new control strategies (IK, neural network,
    reinforcement learning, …).  Each subclass must define
    :meth:`compute_control` and :meth:`is_done`.
    """

    def __init__(self) -> None:
        self._step_counter: int = 0

    @abstractmethod
    def compute_control(self) -> None:
        """Compute one control step."""
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Return ``True`` when the task is complete (success or failure)."""
        ...

    def reset(self) -> None:
        """Reset controller internal state for a new episode."""
        self._step_counter = 0

    @property
    def step_count(self) -> int:
        """Number of times :meth:`compute_control` has been called."""
        return self._step_counter
