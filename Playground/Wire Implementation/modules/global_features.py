from skimage.metrics import structural_similarity as ssim
import numpy as np

def SSIM(orig_img, reconst_img , channel_axis = None,full_state=False,win_size = 11):
    data_range=reconst_img.max() - reconst_img.min()
    return ssim(orig_img, reconst_img , data_range=data_range , win_size=win_size , channel_axis = channel_axis ,full=full_state) 

def PSNR(orig_img, reconst_img):
    mse = np.mean((orig_img - reconst_img) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = max(orig_img.flatten())
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))