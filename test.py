import cv2
import numpy as np
im =cv2.imread("images/test.jpg")


scale_percent = 60 # percent of original size
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)


blur = cv2.pyrMeanShiftFiltering(im,111,151)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

ret ,threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_,cnts,_ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

print(len(cnts))

cv2.drawContours(im,cnts, -1, (0, 255, 0), 3)
cv2.imshow('Display',im)
cv2.waitKey()


