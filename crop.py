# import the necessary packages
import cv2
import numpy as np
from matplotlib import pyplot as plt

# load the image and show it
# img = cv2.imread("/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/src/020219_0.000000_67.000000.jpg")
# cv2.imshow("original", image)
# cv2.waitKey(0)
# img = cv2.imread("/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/src/020219_0.000000_67.000000.jpg")
# skyLine = 90
# crop_img = img[skyLine:, 0:]
# top=0
# bottom=0
# left=60
# right=60
# image = cv2.copyMakeBorder( img, top, bottom, left, right, cv2.BORDER_CONSTANT)


class coordinate:
    def __init__(self, x, y):
        self.x = x
#         self.y = y


# image = cv2.imread(
#     '/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/test.jpg', 1)
# print(image.shape)
# h = image.shape[0]
# w = image.shape[1]
# left_lane = []
# right_lane = []
# zero_image_left_lane = np.zeros((240, 320, 3))
# zero_image_right_lane = np.zeros((240, 320, 3))
# y = 100
# while(y < 140):
#     y += 1
#     for x in range(w):
#         lane_coord = coordinate(x,y)
#         if(image[y, x, 2] > 0):
#             zero_image_left_lane[y, x] = [0, 255, 0]
#             left_lane.append(lane_coord)
#         elif(image[y, x, 1] > 0):
#             zero_image_right_lane[y, x] = [255, 0, 0]
#             right_lane.append(lane_coord)


# zero_image = np.zeros((240, 320, 3))
# lane_follow = []
# # for y in range(100,120,1):
# for l in left_lane:
#     if(100 < l.y < 120 and l.x > 70):
#         for r in right_lane:
#             if(r.y == l.y and r.y < 250):
#                 lf = coordinate(int((l.x+r.x)/2), r.y)
#                 lane_follow.append(lf)
#                 zero_image[lf.y, lf.x] = [255, 255, 255]
#                 zero_image[r.y, r.x] = [255, 0, 0]
#                 zero_image[l.y, l.x] = [0, 255, 0]


# cv2.imshow('image', zero_image)
# cv2.imshow('image2', image)

# cv2.imshow('left', zero_image_left_lane)
# cv2.imshow('right',zero_image_right_lane)
# cv2.waitKey()
# print(x)
# return mask_image
# for i in range(256):
#     for j in range(512):
#         y = []
#         if(embedding_image[i, j, 0] > 0 or embedding_image[i, j, 1] > 0 or embedding_image[i, j, 2] > 0):
#             image_vis[i][j] = [255, 0, 255]
#             # print(i,j)
#             y.append(i)
#             y.append(j)
#             x.append(y)

# lanes = []
# for i in range(5):
#     lane_number = 0
#     x = 0
#     while x < 10:
#         if len(lanes) < lane_number + 1:
#             lanes.append([])
#             lanes[lane_number].append([1, 2])
#             print(lane_number, lanes[0])
#             lane_number += 1
#             x += 1
#         elif lane_number == 4:
#             print('haha')
#         else:
#             x += 1
#     # lanes[lane_number].append([1,2])
#     # lane_number +=1
# img = cv2.imread('/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/test.jpg')
# rows,cols,ch = img.shape

# pts1 = np.float32([[0,40],[320,40],[0,400],[320,400]])
# pts2 = np.float32([[0,0],[320,0],[100,400],[200,400]])

# M = cv2.getPerspectiveTransform(pts1,pts2)

# dst = cv2.warpPerspective(img,M,(300,300))

# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()

zero = np.zeros((250,250,3))
   
# Window name in which image is displayed 
window_name = 'Image'
  
# Start coordinate, here (0, 0) 
# represents the top left corner of image 
start_point = (0, 0) 
  
# End coordinate, here (250, 250) 
# represents the bottom right corner of image 
end_point = (100, 250) 
  
# Green color in BGR 
color = (0, 255, 0) 
  
# Line thickness of 9 px 
thickness = 9
  
# Using cv2.line() method 
# Draw a diagonal green line with thickness of 9 px 
image = cv2.line(zero, start_point, end_point, color, thickness)
cv2.imshow(window_name,zero)
cv2.waitKey()