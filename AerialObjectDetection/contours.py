import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
#value of hue: 
#RGB: 1, 255, 255
#Hex: 01FFFF
#HSV: 180, 100, 100

names = []
area = []
def crop(file):
    original = cv2.imread(file)
    cpy = np.copy(original)

    vectorized = cpy.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    attempts = 11
    img = cv2.cvtColor(cpy, cv2.COLOR_BGR2RGB)
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    #final image
    result_image = res.reshape((img.shape))
    # cv2.imshow("result", result_image)
    # cv2.waitKey(0)

    # need to convert to HSV to obtain a darkline from the blue hue
    imghsv = cv2.cvtColor(result_image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("result_HSV", imghsv)
    # cv2.waitKey(0)

    ##-----------------------------

    ##CONTOUR STUFF

    ###Contour via grayScale
    #gray image from segmentated image
    gray = cv2.cvtColor(imghsv, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # cv2.imshow("tresh", thresh)
    # cv2.waitKey(0)

    #Finds the contours
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # img = cv2.drawContours(original, contours, -1, (0,255,0), 1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    inner = True
    for contour in contours:
        if cv2.contourArea(contour) > 4000:
            epsilon = 0.0001*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            img = cv2.drawContours(original, [approx], 0, (1,1,1), 2)
            if inner:
                print(cv2.contourArea(contour))
                #Black mask
                mask = np.zeros_like(cpy)
                #fills in the mask with the shape of our approximation
                cv2.fillPoly(mask, [approx], color=(255,255,255))
                out = np.zeros_like(original)
                out[mask == 255] = original[mask == 255]
                # cv2.imshow("or", out)
                # cv2.waitKey(0)
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
                area.append(cv2.contourArea(contour))
                # cv2.imwrite(file, cropped_image)
                # cv2.imshow(file, cropped_image)
                # cv2.waitKey(0)
            inner = False


#crop("08-07-353-014_test.png")

# for file in os.listdir('result_images'):
#     crop('result_images/'+file)
#     names.append(file)

# data = pd.DataFrame({'Name':names, 'Area':area})
# data.to_csv('data/detected_area.csv')
