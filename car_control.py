#!/usr/bin/python3
import rospkg
import rospy
import sys
import os
from sensor_msgs.msg import CompressedImage

# try:
#     os.chdir(os.path.dirname(__file__))
#     os.system('clear')
#     print("\nWait for initial setup, please don't connect anything yet...\n")
#     sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
# except:
#     pass

from std_msgs.msg import Float32
import cv2
import numpy as np
from ctypes import *
import math
import random
import time

'''
PARAM WORLD
'''
# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
# width of bottom edge of trapezoid, expressed as percentage of image width
trap_bottom_width = 1
trap_top_width = 1  # ditto for top edge of trapezoid
trap_height = 1  # height of the trapezoid expressed as percentage of image height
sky_line = 75

# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180  # angular resolution in radians of the Hough grid
# minimum number of votes (intersections in Hough grid cell)
threshold = 170
min_line_length = 5  # minimum number of pixels making up a line
max_line_gap = 15    # maximum gap in pixels between connectable line segments

# Dillation
kernel = np.ones((10, 10), np.uint8)
iterations = 1

x_des = 320/2
y_des = 160
car_pos_x = 120
car_pos_y = 300
angle = 0

# Turn param
lost_lane = 0
lost_lane_thresh = 2
turn_left = 0
turn_right = 0
turning_left = 0
turning_right = 0
turn_step_left = 40
turn_step_right = 20

# Hardcode roadblock
rock = 0
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# rospack = rospkg.RosPack()
# path = rospack.get_path('ithutech')
# os.chdir(path)

hasGPU = True
lib = CDLL(
    "./model/darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


altNames = ['turn_left_sign', 'turn_right_sign',
            'rock', 'small_box', 'big_box']
configPath = "./model/yolov3-tiny_obj.cfg"
metaPath = "./model/obj.data"
weightPath = "/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/model/yolov3-tiny_obj_332000.weights"
netMain = load_net_custom(configPath.encode(
    "ascii"), weightPath.encode("ascii"), 0, 1)
metaMain = load_meta(metaPath.encode("ascii"))


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(image, net=netMain, meta=metaMain, thresh=.5, hier_thresh=.5, nms=.45):
    """
    Performs the meat of the detection
    """
    im, _ = array_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    # free_detections(dets, num)
    return res


def to_hls(img):
    """
    Returns the same image in HLS format
    The input image must be in RGB format
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def to_lab(img):
    """
    Returns the same image in LAB format
    Th input image must be in RGB format
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global x_des, y_des, angle, turn_left, turn_right, turning_left, turning_right, rock, lost_lane

    # In case of error, don't draw the line
    draw_right = True
    draw_left = True

    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)

    lines = new_lines

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # x coordinate of center of image
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
        x1, y1, x2, y2 = line[0]

        right_lines_x.append(x1)
        right_lines_x.append(x2)

        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(
            right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
        x1, y1, x2, y2 = line[0]

        left_lines_x.append(x1)
        left_lines_x.append(x2)

        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(
            left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # print('Left:')
    # print('(', left_x1, ',', y1, '), (', left_x2, ',', y2, ')')
    # print('Right:')
    # print('(', right_x1, ',', y1, '), (', right_x2, ',', y2, ')')

    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
    if draw_left and draw_right:
        x_des = int((left_x1 + right_x1)/2)
        y_des = int(y1-50)

    # print('Center:')
    # print('(', x_des, ',', y_des, ')')

    dx = x_des - car_pos_x
    dy = car_pos_y - y_des

    if dx < 0:
        angle = -np.arctan(-dx/dy) * 180/math.pi
    elif dx == 0:
        angle = 0
    else:
        angle = np.arctan(dx/dy) * 180/math.pi

    # size of rock bigger than 4700 (rock is very close)
    if rock >= 4700 and not draw_left and not draw_right:
        turn_left = 0
        turn_right = 0
        angle = 80
    else:
        if turning_left != 0 or turning_right != 0:
            if turning_left > 0 and turning_left < turn_step_left:
                for _ in range(20):
                    car_control(angle=-90, speed=50)
                    print(
                        '------------------------------------Turn leftingg-------------------------------')
                turning_left += 1
            if turning_left == turn_step_left:
                for _ in range(10):
                    car_control(angle=90, speed=20)
                    print(
                        '------------------------------------Correcting-------------------------------')
                for _ in range(30):
                    car_control(angle=-5, speed=50)
                    print(
                        '------------------------------------Correcting-------------------------------')
                turning_left = 0
                turn_left = 0
                turn_right = 0

            if turning_right > 0 and turning_right < turn_step_right:
                car_control(angle=90, speed=20)
                print(
                    '------------------------------------Turn rightingg-------------------------------')
                turning_right += 1
            if turning_right == turn_step_right:
                for _ in range(5):
                    car_control(angle=-90, speed=20)
                    print(
                        '------------------------------------Correcting-------------------------------')
                for _ in range(5):
                    car_control(angle=0, speed=20)
                    print(
                        '------------------------------------Correcting-------------------------------')
                turning_right = 0
                turn_right = 0
                turn_left = 0
        else:
            if not draw_left and not draw_right:
                lost_lane += 1
                if lost_lane > lost_lane_thresh:
                    if turn_left >= 2 or turn_right >= 5:
                        lost_lane = 0
                        for _ in range(1):
                            car_control(angle=0, speed=0)
                        if turn_left - turn_right >= -3:
                            car_control(angle=-90, speed=20)
                            turning_left += 1
                            print(
                                '------------------------------------Turn left-------------------------------')
                        elif turn_right - turn_left >= 5:
                            car_control(angle=90, speed=20)
                            turning_right += 1
                            print(
                                '------------------------------------Turn right------------------------------')
                else:
                    car_control(angle=angle, speed=35)
            else:
                if draw_left and not draw_right:
                    car_control(angle=25, speed=25)
                elif draw_right and not draw_left:
                    car_control(angle=-25, speed=25)
                else:
                    car_control(angle=angle, speed=45)

    img[y_des-5:y_des+5, x_des-5:x_des+5] = (0, 0, 255)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros(img.shape, dtype=np.uint8)  # this produces single-channel (grayscale) image
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
    draw_lines(line_img, lines)
    #draw_lines_debug2(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def abs_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(
        gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
    return gradient_mask


def mag_sobel(gray_img, kernel_size=3, thres=(0, 255)):
    """
    Computes sobel matrix in both x and y directions, merges them by computing the magnitude in both directions
    and applies a threshold value to only set pixels within the specified range
    """
    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sxy = np.sqrt(np.square(sx) + np.square(sy))
    scaled_sxy = np.uint8(255 * sxy / np.max(sxy))

    sxy_binary = np.zeros_like(scaled_sxy)
    sxy_binary[(scaled_sxy >= thres[0]) & (scaled_sxy <= thres[1])] = 1

    return sxy_binary


def combined_sobels(sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=3, angle_thres=(0, np.pi/2)):
    sxy_direction_binary = dir_sobel(
        gray_img, kernel_size=kernel_size, thres=angle_thres)

    combined = np.zeros_like(sxy_direction_binary)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels
    combined[(sx_binary == 1) | ((sy_binary == 1) & (
        sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1

    return combined


def compute_hls_white_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    hls_img = to_hls(rgb_img)

    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hls_img[:, :, 0])
    img_hls_white_bin[((hls_img[:, :, 0] >= 0) & (hls_img[:, :, 0] <= 255))
                      & ((hls_img[:, :, 1] >= 220) & (hls_img[:, :, 1] <= 255))
                      & ((hls_img[:, :, 2] >= 0) & (hls_img[:, :, 2] <= 255))
                      ] = 1

    return img_hls_white_bin


def get_combined_binary_thresholded_img(undist_img):
    """
    Applies a combination of binary Sobel and color thresholding to an undistorted image
    Those binary images are then combined to produce the returned binary image
    """
    undist_img_gray = to_lab(undist_img)[:, :, 0]
    sx = abs_sobel(undist_img_gray, kernel_size=15, thres=(20, 120))
    sy = abs_sobel(undist_img_gray, x_dir=False,
                   kernel_size=15, thres=(20, 120))
    sxy = mag_sobel(undist_img_gray, kernel_size=15, thres=(80, 200))
    sxy_combined_dir = combined_sobels(
        sx, sy, sxy, undist_img_gray, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))

    hls_w_y_thres = compute_hls_white_binary(undist_img)

    combined_binary = np.zeros_like(hls_w_y_thres)
    combined_binary[(sxy_combined_dir == 1) | (hls_w_y_thres == 1)] = 1

    return combined_binary


def dir_sobel(gray_img, kernel_size=3, thres=(0, np.pi/2)):
    """
    Computes sobel matrix in both x and y directions, gets their absolute values to find the direction of the gradient
    and applies a threshold value to only set pixels within the specified range
    """
    sx_abs = np.absolute(
        cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sy_abs = np.absolute(
        cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))

    dir_sxy = np.arctan2(sx_abs, sy_abs)

    binary_output = np.zeros_like(dir_sxy)
    binary_output[(dir_sxy >= thres[0]) & (dir_sxy <= thres[1])] = 1

    return binary_output



# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_out_name = 'output_{}.avi'.format(time.time())
# video_out = cv2.VideoWriter(video_out_name, fourcc, 20, (320, 240))

'''
PARAM WORLD
'''


def car_control(angle=angle, speed=50):
    pub_speed = rospy.Publisher('/ithutech_speed', Float32, queue_size=10)
    pub_angle = rospy.Publisher('/ithutech_steerAngle', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)


def image_callback(data):
    global turn_left, turn_right, rock
    start_time = time.time()
    temp = np.fromstring(data.data, np.uint8)
    img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    frame = img

    # Lane detect
    frame = frame[sky_line:, :]
    frame = cv2.dilate(frame, kernel, iterations=iterations)
    combined = get_combined_binary_thresholded_img(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) * 255
    line_image = hough_lines(combined, rho, theta,
                             threshold, min_line_length, max_line_gap)
    test_img = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    annotated_image = cv2.cvtColor(weighted_img(
        line_image, test_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('ithutech_lane_detection', annotated_image)

    # # Object detect
    # detections = detect(image=img, thresh=0.25)
    # if detections:
    #     for each_detection in detections:
    #         print('{}: {}%'.format(each_detection[0], each_detection[1]*100))
    #         if each_detection[0] == 'turn_left_sign':
    #             turn_left += 1
    #         if each_detection[0] == 'turn_right_sign':
    #             turn_right += 1
    #         x_center = each_detection[-1][0]
    #         y_center = each_detection[-1][1]
    #         width = each_detection[-1][2]
    #         height = each_detection[-1][3]
    #         x_top = int(x_center - width/2)
    #         y_top = int(y_center - height/2)
    #         x_bot = int(x_top + width)
    #         y_bot = int(y_top + height)
    #         if each_detection[0] == 'rock':
    #             rock = width * height
    #             print('Size: ', rock)
    #         cv2.rectangle(img, (x_top, y_top), (x_bot, y_bot), (0, 255, 0), 2)

    # video_out.write(annotated_image)
    cv2.imshow('ithutech_object_detector', img)
    cv2.waitKey(1)
    print('FPS:', 1/(time.time() - start_time))


