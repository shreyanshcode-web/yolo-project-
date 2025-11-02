import os
import numpy as np
import cv2
os.makedirs('Midas/outputs/depth', exist_ok=True)
os.makedirs('Midas/outputs/rgb', exist_ok=True)
# create dummy depth image (grayscale)
depth = np.full((240,320), 128, dtype='uint8')
cv2.imwrite('Midas/outputs/depth/frame.png', depth)
# create dummy rgb image
rgb = np.zeros((240,320,3), dtype='uint8')
cv2.circle(rgb, (160,120), 50, (0,255,0), -1)
cv2.imwrite('Midas/outputs/rgb/frame.png', rgb)
print('Wrote dummy images')
