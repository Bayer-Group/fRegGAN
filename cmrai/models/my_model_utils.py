import random

import numpy as np
import torch
import torch.nn.functional as F


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if (
                self.num_imgs < self.pool_size
            ):  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if (
                    p > 0.5
                ):  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(
                        0, self.pool_size - 1
                    )  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


def set_requires_grad(nets, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def smoothing_loss(y_pred, penalty="l2"):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    if penalty == "l2":
        dx = dx * dx
        dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d / 2
    return grad


def create_circular_mask(h, w, center=None, radius=None):
    # https://stackoverflow.com/a/44874588
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def fft2d_loss(real_B, fake_B, r, freq_weighting, use_euclid=False):
    fft_real_B = torch.fft.fftshift(torch.fft.fft2(real_B))
    fft_fake_B = torch.fft.fftshift(torch.fft.fft2(fake_B))

    if use_euclid:
        loss_fct = fft2d_euclid
    else:
        fft_real_B = torch.abs(fft_real_B)
        fft_fake_B = torch.abs(fft_fake_B)
        loss_fct = F.l1_loss

    mask_low_freq = create_circular_mask(*fft_real_B.shape[-2:], radius=r)
    fft_real_B_low = fft_real_B[:, :, mask_low_freq]
    fft_fake_B_low = fft_fake_B[:, :, mask_low_freq]

    fft_real_B_high = fft_real_B[:, :, ~mask_low_freq]
    fft_fake_B_high = fft_fake_B[:, :, ~mask_low_freq]

    low_freq_loss = loss_fct(fft_fake_B_low, fft_real_B_low)
    high_freq_loss = loss_fct(fft_fake_B_high, fft_real_B_high)
    return freq_weighting * high_freq_loss + (1 - freq_weighting) * low_freq_loss


def fft2d_euclid(fake_img, real_img):
    fake_freq = torch.stack([fake_img.real, fake_img.imag], -1)
    real_freq = torch.stack([real_img.real, real_img.imag], -1)
    tmp = (fake_freq - real_freq) ** 2
    freq_distance = tmp[..., 0] + tmp[..., 1]  # keine sqrt weil **2 sqt distance
    return torch.mean(freq_distance)
