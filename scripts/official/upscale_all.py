import cv2
import numpy as np
import matplotlib.pyplot as plt

for k in range(31):
    f_in  = f"downscaled/{k:02d}.png"
    f_out  = f"upscaled_bicubic/{k:02d}.png"
    print(f_in, f_out)

    img = cv2.imread(f_in)
    # resize to 256x256
    img_bis = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f_out, img_bis)


for k in range(31):
    f_in  = f"downscaled/{k:02d}.png"
    f_out  = f"upscaled_nearest/{k:02d}.png"
    print(f_in, f_out)

    img = cv2.imread(f_in)
    # resize to 256x256
    img_bis = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imwrite(f_out, img_bis)
