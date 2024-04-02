from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn
from .visualizer import Visualizer

__all__ = ["VisualizerCollection"]


class VisualizerCollection():
    def __init__(
        self,
        visualizers: List[Visualizer] = []
    ) -> None:
        super().__init__()
        self.visualizers = visualizers
