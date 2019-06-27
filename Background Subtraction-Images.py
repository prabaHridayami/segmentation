import numpy as np
import cv2

img1 = cv2.imread('citra-background-rgb.jpg') #open citra 1
img2 = cv2.imread('citra-foreground-rgb.jpg') # open citra 2
#deklarasi fungsi MOG, MOG2 & GMG
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

fgmask11 = mog.apply(img1) # Fungsi MOG citra 1
fgmask12 = mog.apply(img2) # Fungsi MOG citra 2
fgmask21 = mog2.apply(img1) # Fungsi MOG2 citra 1
fgmask22 = mog2.apply(img2) # Fungsi MOG2 citra 2
fgmask31 = gmg.apply(img1) # Fungsi GMG citra 1
fgmask32 = gmg.apply(img2) # Fungsi GMG citra 2
fgmask31 = cv2.morphologyEx(fgmask31, cv2.MORPH_OPEN, kernel)
fgmask32 = cv2.morphologyEx(fgmask32, cv2.MORPH_OPEN, kernel)

# Tampilkan Citra Hasil
cv2.imshow('Background', cv2.resize(img1, (400, 300)))
cv2.imshow('Foreground', cv2.resize(img2, (400, 300)))
cv2.imshow('MOG', cv2.resize(fgmask12, (400, 300)))
cv2.imshow('MOG2', cv2.resize(fgmask22, (400, 300)))
cv2.imshow('GMG', cv2.resize(fgmask32, (400, 300)))

cv2.waitKey(0)
