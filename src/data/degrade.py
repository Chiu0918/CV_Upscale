import numpy as np
import cv2


def official_downscale(img: np.ndarray) -> np.ndarray:
    """
    Downscale an HR image (e.g. 256x256) to LR (e.g. 64x64)
    using the exact official method: img[::4, ::4, :]

    Parameters:
        img (np.ndarray): Input image in shape (H, W, C)

    Returns:
        np.ndarray: Downscaled image in shape (H/4, W/4, C)
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    return img[::4, ::4, :]


def load_image(path: str) -> np.ndarray:
    """
    Load an image from the specified path using OpenCV.

    Parameters:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a numpy array (H, W, C), BGR order.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def save_image(img: np.ndarray, path: str):
    """
    Save a numpy array as an image to the specified path using OpenCV.

    Parameters:
        img (np.ndarray): Image data as a numpy array (H, W, C).
        path (str): Path to save the image file.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Image to save must be a numpy array.")
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")