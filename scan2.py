# USAGE
# python scan.py --image images/page.jpg
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import histogram as h
import cumulative_histogram as ch

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


for c in cnts:
	
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		screenCnt = approx
		break


print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
height = warped.shape[0]
width = warped.shape[1]
pixels = width * height

hist = h.histogram(warped)
cum_hist = ch.cumulative_histogram(hist)
brightness = 40
p = 0.005

a_low = 0
for i in np.arange(256):
    if cum_hist[i] >= pixels * p:
        a_low = i
        break
    
a_high = 255
for i in np.arange(255, -1, -1):
    if cum_hist[i] <= pixels * (1 - p):
        a_high = i
        break
  
for i in np.arange(height):
    for j in np.arange(width):
        a = warped.item(i,j)
        b = 0
        if a <= a_low:
            b = 0
        elif a >= a_high:
            b = 255
        else:
            b = float(a - a_low) / (a_high - a_low) * 255
        warped.itemset((i,j), b)

for i in np.arange(height):
    for j in np.arange(width):
        a = warped.item(i,j)
        b = a + brightness
        if b > 255:
            b = 255
        warped.itemset((i,j), b)

T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
