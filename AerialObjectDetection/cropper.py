import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#value of hue: 
#RGB: 1, 255, 255
#Hex: 01FFFF
#HSV: 180, 100, 100

name = '08-07-352-007.png'
original = cv2.imread('images/'+name)
cpy = np.copy(original)

#converts the image to transparent png
def blackToTransparent(img):

    src = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    bgr = src[:,:,:3] # Channels 0..2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Some sort of processing...

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = src[:,:,3] # Channel 3
    result = np.dstack([bgr, alpha]) # Add the alpha channel
    cv2.imshow("test?", result)
    cv2.waitKey(0)


    # cv2.imwrite('51IgH_result.png', result)

##Kmeans image segmentation
vectorized = cpy.reshape((-1, 3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts = 10
img = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)
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

##CONTOUR STUFF

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
inner = True
for contour in contours:
    if cv2.contourArea(contour) > 4000:
        epsilon = 0.0001*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(cpy, [approx], 0, (0), 1)
        x,y = approx[0][0]
        if inner:
            #Black mask
            mask = np.zeros_like(cpy)
            #fills in the mask with the shape of our approximation
            cv2.fillPoly(mask, [approx], color=(255,255,255))
            out = np.zeros_like(original)
            out[mask == 255] = original[mask == 255]
            cv2.imshow("or", out)
            cv2.waitKey(0)
            #study masks!!!
            #https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour

            #cropping edges
            ext_left = tuple(approx[approx[:, :, 0].argmin()][0])
            ext_right = tuple(approx[approx[:, :, 0].argmax()][0])
            ext_top = tuple(approx[approx[:, :, 1].argmin()][0])
            ext_bot = tuple(approx[approx[:, :, 1].argmax()][0])
            #roi == approx
            #Cropping
            cropped_image = out[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
            # cv2.imwrite('crop{}.jpg'.format('_'+str(1)), cropped_image)
            # cv2.imwrite('{}'.format(name),cropped_image)
            # cv2.imshow("final?", cropped_image)
            # cv2.waitKey(0)

            

        inner = False


blackToTransparent(name)
