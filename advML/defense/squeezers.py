import torch
import numpy as np
from scipy import ndimage
import skimage.restoration as restoration


def bit_depth_squeeze(X: torch.Tensor, bit_depth: int) -> torch.Tensor:
    """Reduces the color bit depth of an image or a set of images.

    :param X: a float tensor of image(s) which have been scaled to [0, 1]
    :type X: torch.Tensor
    :param bit_depth: the color bit depth to reduce the image to
    :type bit_depth: int
    :return: a float tensor of reduced image(s)
    :rtype: torch.Tensor
    """
    precision = (2**bit_depth) - 1
    X_squeezed = torch.round(X * precision)
    X_squeezed /= precision
    return X_squeezed


# TODO implement using PyTorch utilities instead
def median_filter_squeeze(X: torch.Tensor, size: int) -> torch.Tensor:
    """Wrapper for scipy's median_filter.
    Applies a local median filter to an image or set of images.

    :param X: a float tensor of image(s)
    :type X: torch.Tensor
    :param size: the size of the sliding window for the median filter
    :type size: int
    :return: a float tensor of smoothed image(s)
    :rtype: torch.Tensor
    """
    if X.ndim < 2 or X.ndim > 4:
        AssertionError(f"Expected a 2D, 3D, or 4D tensor. \
                       Received a tensor of dim {X.ndim}")
    X_np = X.detach().cpu().numpy()

    if X_np.ndim == 2:
        X_np = ndimage.median_filter(X_np,
                                     size=(size, size),
                                     mode="reflect")
    elif X_np.ndim == 3:
        X_np = ndimage.median_filter(X_np,
                                     size=(1, size, size),
                                     mode="reflect")
    elif X_np.ndim == 4:
        X_np = ndimage.median_filter(X_np,
                                     size=(1, 1, size, size),
                                     mode="reflect")

    X_squeezed = torch.from_numpy(X_np).to(X.device)
    return X_squeezed


# TODO implement using PyTorch utilities instead
def mean_filter_squeeze(X: torch.Tensor, size: int) -> torch.Tensor:
    """Wrapper for scipy's uniform_filter.
    Applies a local mean filter to an image or a set of images.

    :param X: a float tensor of image(s)
    :type X: torch.Tensor
    :param size: size of the sliding window for the mean filter
    :type size: int
    :return: a float tensor of smoothed image(s)
    :rtype: torch.Tensor
    """
    if X.ndim < 2 or X.ndim > 4:
        AssertionError(f"Expected a 2D, 3D, or 4D tensor. \
                       Received a tensor of dim {X.ndim}")
    X_np = X.detach().cpu().numpy()

    if X_np.ndim == 2:
        X_np = ndimage.uniform_filter(X_np,
                                      size=(size, size),
                                      mode="reflect")
    elif X_np.ndim == 3:
        X_np = ndimage.uniform_filter(X_np, size=(1, size, size),
                                      mode="reflect")
    elif X_np.ndim == 4:
        X_np = ndimage.uniform_filter(X_np, size=(1, 1, size, size),
                                      mode="reflect")

    X_squeezed = torch.from_numpy(X_np).to(X.device)
    return X_squeezed


# TODO implement using PyTorch utilities instead
def non_local_means_squeeze(X: torch.Tensor,
                            patch_size: int, patch_distance: int,
                            fast_mode: bool) -> torch.Tensor:
    """Wrapper for scikit-image's denoise_nl_means.
    Applies non-local means denoising on an image or a set of images.

    :param X: a float tensor of image(s)
    :type X: torch.Tensor
    :param patch_size: size of patches for denoising
    :type patch_size: int
    :param patch_distance: distance in pixels to search for patches used for denoising
    :type patch_distance: int
    :param fast_mode: whether to use fast version of non-local means algorithm 
    (defaults to True for 3D and 4D tensors)
    :type fast_mode: bool
    :return: a float tensor of denoised image(s)
    :rtype: torch.Tensor
    """
    if X.ndim < 2 or X.ndim > 4:
        AssertionError(f"Expected a 2D, 3D, or 4D tensor. \
                       Received a tensor of dim {X.ndim}")
    X_np = X.detach().cpu().numpy()

    patch_kwargs = dict(patch_size=patch_size,
                        patch_distance=patch_distance,
                        channel_axis=-1)
    if X.ndim == 2:
        X_np = restoration.denoise_nl_means(X_np,
                                            fast_mode=fast_mode,
                                            **patch_kwargs)
    elif X.ndim == 3:
        X_np = restoration.denoise_nl_means(X_np.transpose(1, 2, 0),
                                            fast_mode=True,
                                            **patch_kwargs).transpose(2, 0, 1)
    elif X.ndim == 4:
        for i in range(X_np.shape[0]):
            X_np[i] = restoration.denoise_nl_means(X_np[i].transpose(1, 2, 0),
                                                   fast_mode=True,
                                                   **patch_kwargs).transpose(2, 0, 1)

    X_squeezed = torch.from_numpy(X_np).to(X.device)
    return X_squeezed
