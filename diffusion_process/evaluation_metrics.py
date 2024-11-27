from torchmetrics.image.fid import FrechetInceptionDistance
import torch as th
from multi_modal_diffusion import dist_util


#  FOR IMAGE

def compute_fid( generated_images, real_images):
    # Ensure images are in the range [0, 1]
    generated_images = (generated_images + 1) / 2  # Assuming original range is [-1, 1]
    real_images = (real_images + 1) / 2

    # Handle grayscale images by repeating channels
    if generated_images.shape[1] == 1:
        generated_images = generated_images.repeat(1, 3, 1, 1)
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)

    # Resize images to 299x299 (required by Inception network)
    generated_images = th.nn.functional.interpolate(generated_images, size=(299, 299), mode='bilinear')
    real_images = th.nn.functional.interpolate(real_images, size=(299, 299), mode='bilinear')

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048).to(dist_util.dev())

    # Update FID with real and generated images
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    # Compute FID score
    fid_score = fid.compute()
    return fid_score.item()


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (gamma * X @ Y.T + coef0) ** degree
    return K


def compute_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]

    # Flatten images
    X = X.view(m, -1)
    Y = Y.view(n, -1)

    # Compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)

    # Compute MMD
    K_XX_sum = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)

    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum

    return mmd.item()


#  FOR TABULAR

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    dist = th.cdist(X, Y, p=2) ** 2
    K = th.exp(-gamma * dist)
    return K


def compute_mmd_tabular( X, Y):
    m = X.shape[0]
    n = Y.shape[0]

    # Compute kernels
    K_XX = rbf_kernel(X)
    K_YY = rbf_kernel(Y)
    K_XY = rbf_kernel(X, Y)

    # Compute MMD
    K_XX_sum = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)

    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum

    return mmd.item()
