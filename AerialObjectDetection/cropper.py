import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

original = cv2.imread('images/08-07-352-007.png')

##Kmeans image segmentation
vectorized = original.reshape((-1, 3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts = 10
img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
#final image
result_image = res.reshape((img.shape))
cv2.imshow("result", result_image)
cv2.waitKey(0)

# need to convert to HSV to obtain a darkline from the blue hue
imghsv = cv2.cvtColor(result_image, cv2.COLOR_BGR2HSV)
cv2.imshow("result_HSV", imghsv)
cv2.waitKey(0)

##-----------------------------

###Contour via grayScale
#gray image from segmentated image
gray = cv2.cvtColor(imghsv, cv2.COLOR_RGB2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)

ret, thresh = cv2.threshold(gray, 127, 255, 0)
cv2.imshow("tresh", thresh)
cv2.waitKey(0)

#Finds the contours
img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#processes the contours and recrops the image to the property size
counter = 0
for contour in contours:
    if cv2.contourArea(contour) > 4000:
        epsilon = 0.01*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(original, [approx], 0, (0), 3)
        x,y = approx[0][0]

        cv2.imshow("or", original)
        cv2.waitKey(0)

        # counter += 1
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        # ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        # ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        # ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])
        # roi_corners = np.array([box], dtype=np.int32)

        # cv2.polylines(img, roi_corners, 1, (255, 0, 0), 3)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cropped_image = original[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        # cv2.imwrite('crop{}.jpg'.format('_'+str(counter)), cropped_image)


##Contour Via HSV
#value of hue: 
#RGB: 1, 255, 255
#Hex: 01FFFF
#HSV: 180, 100, 100
# hue = np.array([1,255,255])
# imghsv = cv2.cvtColor(cpy, cv2.COLOR_BGR2HSV)

# cv2.imshow("HSV", imghsv)
# cv2.waitKey(0)
# mask = cv2.inRange(imghsv, hue, hue)

# fimg, contours, hierarchy2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(cpy, contours, -1, (255,0,0), 8)
# cv2.imshow("HSV", cpy)
# cv2.waitKey(0)
