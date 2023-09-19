import numpy as np


def psnr_masked(fake, real, background=0):
    output = []
    for idx in range(fake.shape[0]):
        mse = np.mean(
            (fake[idx][real[idx] != background] - real[idx][real[idx] != background])
            ** 2
        )
        output += [100 if mse < 1.0e-10 else 20 * np.log10(1 / np.sqrt(mse))]
    return output


def mae_masked(fake, real, background=0):
    output = []
    for idx in range(fake.shape[0]):
        mae = np.mean(
            np.abs(
                fake[idx][real[idx] != background] - real[idx][real[idx] != background]
            )
        )
        output += [mae]
    return output


def mae(fake, real):
    output = []
    for idx in range(fake.shape[0]):
        mae = np.mean(np.abs(fake[idx] - real[idx]))
        output += [mae]
    return output
