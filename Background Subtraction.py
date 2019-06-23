import numpy as np
import cv2

cap = cv2.VideoCapture('tes1.mp4') #deklarasi video
#deklarasi fungsi MOG, MOG2 & GMG
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()  # membaca input
    #Tampilkan Video Asli
    cv2.imshow('Original', cv2.resize(frame, (400, 300)))
    fgmask1 = mog.apply(frame)  # Fungsi MOG
    fgmask2 = mog2.apply(frame) #Fungsi MOG2
    fgmask3 = gmg.apply(frame)  # Fungsi GMG
    fgmask3 = cv2.morphologyEx(fgmask3, cv2.MORPH_OPEN, kernel)

    #Tampilkan Hasil
    cv2.imshow('MOG', cv2.resize(fgmask1, (400, 300)))
    cv2.imshow('MOG2', cv2.resize(fgmask2, (400, 300)))
    cv2.imshow('GMG', cv2.resize(fgmask3, (400, 300)))
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break   
cap.release()
cv2.destroyAllWindows()
