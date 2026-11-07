"""Constraints g_{i,k}(a_i, a_{-i}, theta) <= 0.

kind = "individual": depends only on (a_i, theta_i); no expectation needed.
kind = "expected":   h_{i,k} = E_{theta_{-i}}[g_{i,k}(a_i, f_{-i}(theta_{-i}), theta)],
                     estimated by Monte Carlo or quadrature.
Box bounds are individual constraints.
"""
from dataclasses import dataclass
from typing import Callable, Literal

import torch

# g(a_i, a_opp, theta_i, theta_opp) -> tensor of shape (batch,)
ConstraintFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class Constraint:
    name: str
    kind: Literal["individual", "expected"]
    fn: ConstraintFn
