import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('lena-1.png')     # input
mask = cv2.imread('lena-2.png',0)  # mask

dst_TELEA = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
dst_NS = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('degraded image')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.title('mask image')
plt.subplot(223), plt.imshow(cv2.cvtColor(dst_TELEA, cv2.COLOR_BGR2RGB))
plt.title('TELEA')
plt.subplot(224), plt.imshow(cv2.cvtColor(dst_NS, cv2.COLOR_BGR2RGB))
plt.title('NS')

plt.tight_layout()
plt.show()