"""Game interface.

A game states only its primitives. The merit (gbne.merit) is built generically from
these, with gradients d/da_i taken by autodiff.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from gbne.games.constraints import Constraint


class Game(ABC):
    n_players: int
    type_dims: Tuple[int, ...]    # d_i
    action_dims: Tuple[int, ...]  # z_i

    @abstractmethod
    def sample_types(self, i: int, n: int, generator: torch.Generator = None) -> torch.Tensor:
        """n independent draws of theta_i, shape (n, d_i)."""

    @abstractmethod
    def utility(self, i: int, a_i: torch.Tensor, a_opp: List[torch.Tensor],
                theta_i: torch.Tensor, theta_opp: List[torch.Tensor]) -> torch.Tensor:
        """u_i(a, theta), shape (batch,)."""

    @abstractmethod
    def action_box(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(lower, upper) bounds of A_i used by the strategy head."""

    def constraints(self, i: int) -> List[Constraint]:
        """Constraints of player i. Default: none (plain BNE)."""
        return []
