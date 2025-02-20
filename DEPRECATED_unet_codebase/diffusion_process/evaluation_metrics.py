from torchmetrics.image.fid import FrechetInceptionDistance
import torch as th
from multi_modal_diffusion import dist_util


#  FOR IMAGE

# def compute_fid( generated_images, real_images):
#     # Ensure images are in the range [0, 1]
#     generated_images = (generated_images + 1) / 2  # Assuming original range is [-1, 1]
#     real_images = (real_images + 1) / 2
#
#     # Handle grayscale images by repeating channels
#     if generated_images.shape[1] == 1:
#         generated_images = generated_images.repeat(1, 3, 1, 1)
#     if real_images.shape[1] == 1:
#         real_images = real_images.repeat(1, 3, 1, 1)
#
#     # Resize images to 299x299 (required by Inception network)
#     generated_images = th.nn.functional.interpolate(generated_images, size=(299, 299), mode='bilinear')
#     real_images = th.nn.functional.interpolate(real_images, size=(299, 299), mode='bilinear')
#
#     # Initialize FID metric
#     fid = FrechetInceptionDistance(feature=2048).to(dist_util.dev())
#
#     # Update FID with real and generated images
#     fid.update(real_images, real=True)
#     fid.update(generated_images, real=False)
#
#     # Compute FID score
#     fid_score = fid.compute()
#     return fid_score.item()

def compute_fid(generated_images, real_images):

    # Ensure images are on the same device
    device = generated_images.device

    # Initialize FID metric on the correct device and disable synchronization
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Update with real and generated images
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


# def compute_mmd(X, Y):
#     m = X.shape[0]
#     n = Y.shape[0]
#
#     # Flatten images
#     X = X.view(m, -1)
#     Y = Y.view(n, -1)
#
#     # Compute kernels
#     K_XX = polynomial_kernel(X)
#     K_YY = polynomial_kernel(Y)
#     K_XY = polynomial_kernel(X, Y)
#
#     # Compute MMD
#     K_XX_sum = (K_XX.sum() - K_XX.diag().sum()) / (m * (m - 1))
#     K_YY_sum = (K_YY.sum() - K_YY.diag().sum()) / (n * (n - 1))
#     K_XY_sum = K_XY.sum() / (m * n)
#
#     mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
#
#     return mmd.item()

def compute_mmd(generated_images, real_images):
    # Ensure images are of shape [N, C, H, W]
    assert generated_images.shape == real_images.shape, "Generated and real images must have the same shape"

    # Flatten images if required
    N = generated_images.size(0)
    generated_images_flat = generated_images.view(N, -1)
    real_images_flat = real_images.view(N, -1)

    # Compute MMD using a Gaussian kernel
    mmd_value = mmd_rbf(generated_images_flat, real_images_flat)
    return mmd_value.item()


def mmd_rbf(X, Y, sigma=1.0):
    XX = th.matmul(X, X.t())
    YY = th.matmul(Y, Y.t())
    XY = th.matmul(X, Y.t())

    rx = (XX.diag().unsqueeze(0).expand_as(XX))
    ry = (YY.diag().unsqueeze(0).expand_as(YY))

    K = th.exp(- (rx.t() + rx - 2 * XX) / (2 * sigma ** 2))
    L = th.exp(- (ry.t() + ry - 2 * YY) / (2 * sigma ** 2))
    P = th.exp(- (rx.t() + ry - 2 * XY) / (2 * sigma ** 2))

    beta = 1. / (X.size(0) * X.size(0))
    gamma = 1. / (Y.size(0) * Y.size(0))
    delta = 2. / (X.size(0) * Y.size(0))

    mmd = beta * K.sum() + gamma * L.sum() - delta * P.sum()
    return mmd


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
