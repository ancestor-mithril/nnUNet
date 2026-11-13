"""
Deprecated location. ResEncUNetPlanner moved to
nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners,
where it lives next to the nnUNetPlannerResEncM/L/XL presets that derive from it.

This module used to hold a second, independently maintained copy of the class. Because `-pl` resolves
planners by name and returns the first match while walking the package tree, `-pl ResEncUNetPlanner`
picked up this stale copy rather than the maintained one, while the M/L/XL presets used the maintained
one - so the bare planner and its own presets could disagree. It is now a re-export, so both paths give
the same class.
"""
import warnings

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    ResEncUNetPlanner,
)

__all__ = ['ResEncUNetPlanner']

warnings.warn(
    "nnunetv2.experiment_planning.experiment_planners.resencUNet_planner is deprecated. Import "
    "ResEncUNetPlanner from nnunetv2.experiment_planning.experiment_planners.residual_unets."
    "residual_encoder_unet_planners instead.",
    DeprecationWarning,
    stacklevel=2,
)
