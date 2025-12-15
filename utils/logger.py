import copy

import numpy as np
from torch_cubic_spline_grids import CubicBSplineGrid3d


class LogIO:
    def __init__(self, func):
        self.func = func
        self.evaluated_inputs = []
        self.evaluated_outputs = []

    def __call__(self, arg):
        self.log_inputs(arg)
        result = self.func(arg)
        self.log_outputs(result)
        return result

    def get_logged_inputs(self):
        return self.evaluated_inputs

    def get_logged_outputs(self):
        return self.evaluated_outputs

    def log_inputs(self, arg):
        if isinstance(arg, CubicBSplineGrid3d):
            self.evaluated_inputs.append(copy.deepcopy(arg))
        if isinstance(arg, np.ndarray):
            self.evaluated_inputs.append(arg)

    def log_outputs(self, result):
        if isinstance(result, tuple):
            self.evaluated_outputs.append(
                (result[0].detach().item(), result[1].detach().item())
            )
        if isinstance(result, np.floating):
            self.evaluated_outputs.append(result)
