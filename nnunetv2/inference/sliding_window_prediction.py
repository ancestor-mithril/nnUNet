import os
from functools import lru_cache

import numpy as np
import torch
from typing import Union, Tuple, List
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter

from nnunetv2.utilities.logging import perf_logger


@lru_cache(maxsize=0 if os.getenv("NO_HEATMAP_CACHE", "0") == "1" else None)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    use_gaussian = os.getenv("USE_GAUSSIAN", "0") == "1"
    print("Using gaussian" if use_gaussian else "Using hanning")
    if not use_gaussian:
        # 1D Hanning window for each dimension
        windows_1d = [np.hanning(sz).astype(np.float16) for sz in tile_size]

        # Create multi-dimensional Hanning map via meshgrid
        grids = np.meshgrid(*windows_1d, indexing='ij')
        del windows_1d
        hanning_map = np.prod(grids, axis=0)
        del grids

        # Scale max value for numerical stability
        hanning_map /= (np.max(hanning_map) / value_scaling_factor)

        # Convert to torch tensor
        hanning_map = torch.from_numpy(hanning_map).to(device=device, dtype=dtype)

        # Avoid zeros (important to prevent NaNs during division)
        mask = hanning_map == 0
        if mask.any():
            hanning_map[mask] = torch.min(hanning_map[~mask])
        perf_logger.info(f"Finished building Hanning {hanning_map.shape}")
        del mask
        return hanning_map
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


if __name__ == '__main__':
    a = torch.rand((4, 2, 32, 23))
    a_npy = a.numpy()

    a_padded = pad_nd_image(a, new_shape=(48, 27))
    a_npy_padded = pad_nd_image(a_npy, new_shape=(48, 27))
    assert all([i == j for i, j in zip(a_padded.shape, (4, 2, 48, 27))])
    assert all([i == j for i, j in zip(a_npy_padded.shape, (4, 2, 48, 27))])
    assert np.all(a_padded.numpy() == a_npy_padded)
