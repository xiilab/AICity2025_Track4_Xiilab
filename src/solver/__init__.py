"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Dict

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .robust_solver import RobustSolver

TASKS: Dict[str, BaseSolver] = {
    "classification": ClasSolver,
    "detection": DetSolver,
    "robust": RobustSolver,
}
