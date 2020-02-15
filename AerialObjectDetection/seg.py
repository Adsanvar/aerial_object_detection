import numpy as np
import cv2
import matplotlib.pyplot as plt

#image = cv2.imread('dashcam.png')
image = cv2.imread('house_export.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1, 3))

vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 4
attempts = 10

sse = []

##Kmeans Elbow chart
# for i in range(1,11):
#     ret, label, center = cv2.kmeans(vectorized, i, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
#     sse.append(ret)

# fig  = plt.figure()
# plt.plot(sse)
# plt.title("Elbow Kmeans")
# plt.xlabel('K')
# plt.ylabel('SSE')
# fig.savefig('test.png', format= 'png')
# plt.show()

##Iterations for various kmeans
# for i in range(2,7):
#     ret, label, center = cv2.kmeans(vectorized, i, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

#     center = np.uint8(center)

#     res = center[label.flatten()]

#     result_image = res.reshape((img.shape))

#     fig1 = plt.figure()
#     plt.subplot(1,2,1), plt.imshow(img)
#     plt.title("Original"), plt.xticks([]),plt.yticks([])
#     plt.subplot(1,2,2),plt.imshow(result_image)
#     plt.title("Segmented Image When K = %i" % i), plt.xticks([]), plt.yticks([])
#     fig1.savefig('segmented_Auto_k' + str(i) +'.png', format= 'png')
#     #plt.show()

# ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

# center = np.uint8(center)

# res = center[label.flatten()]

# result_image = res.reshape((img.shape))

gray = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 1, 100, apertureSize = 3)

# fig1 = plt.figure()
# plt.subplot(1,2,1), plt.imshow(result_image)
# plt.title("Kmeans Segmentation"), plt.xticks([]),plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(edges, cmap='gray')
# plt.title("Edge Detection" ), plt.xticks([]), plt.yticks([])
# fig1.savefig('EDGE.png', format= 'png')
# #plt.show()

