import cv2
import numpy as np
import matplotlib.pyplot as plt

for k in range(31):
    f_in  = f"originals_bis/{k:02d}.png"
    f_out  = f"downscaled/{k:02d}.png"
    print(f_in, f_out)

    img = cv2.imread(f_in)
    img_bis = img[::4,::4,:]
    cv2.imwrite(f_out, img_bis)
