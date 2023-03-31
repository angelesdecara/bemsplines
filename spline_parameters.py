from dataclasses import dataclass

import numpy as np

@dataclass
class spline_parameters:
    number_of_knots: int
    interpolation_density: int
    minimum_derivatives_cost_reduction: float
    minimum_overall_cost_reduction: float
    projection_interpolation_density: int
    torso_potentials: np.ndarray
