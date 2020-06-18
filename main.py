#!/usr/bin/python3
import argparse
import os.path as ops
import glog as log
import matplotlib.pyplot as plt
import tensorflow as tf
from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import scipy.misc
import time
import random
import math
from ctypes import *
import numpy as np
import cv2
from std_msgs.msg import Float32
import rospkg
import rospy
import sys
import os
from sensor_msgs.msg import CompressedImage
from skimage import io, draw
import threading
import time
# import gi
# gi.require_version('Gtk', '2.0')
try:
    os.chdir(os.path.dirname(__file__))
    os.system('clear')
    print("\nWait for initial setup, please don't connect anything yet...\n")
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except:
    pass
CFG = global_config.cfg

turn_right = turn_left = False
lane_detect = True
frame = 0
image_0 = None
image_object = None
speed = 50
angle = 0
car_x = 160
skyLine = 80
wait_to_turn = False
t_start = None
have_data = False
detect_object_alive = False
detect_lane_alive = False
avoid_object = False
br = False


srcPath = '/home/ntl/Downloads/py_digitalrace_2019-master/src/chickenteam/src/'
weights_path = srcPath + 'model/tusimple_lanenet_vgg.ckpt'
image_path = '/home/ntl/test.jpg'
configPath = "./model/yolov3-tiny_obj.cfg"
metaPath = "./model/obj.data"
weightPath = "./model/yolov3-tiny_obj_332000.weights"
modelPath = "./model/darknet.so"
altNames = None
if altNames is None:
    # In Python 3, the metafile default access craps out on Windows (but not Linux)
    # Read the names file and create a list to feed to detect
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


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
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


rospack = rospkg.RosPack()
path = rospack.get_path('ithutech')
os.chdir(path)

hasGPU = True
lib = CDLL(modelPath, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

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

network_predict = lib.network_predict_ptr
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

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

netMain = load_net_custom(configPath.encode(
    "ascii"), weightPath.encode("ascii"), 0, 1)
metaMain = load_meta(metaPath.encode("ascii"))

with open(metaPath) as metaFH:
    metaContents = metaFH.read()
    import re
    match = re.search("names *= *(.*)$", metaContents,
                      re.IGNORECASE | re.MULTILINE)
    if match:
        result = match.group(1)
    else:
        result = None
    try:
        if os.path.exists(result):
            with open(result) as namesFH:
                namesList = namesFH.read().strip().split("\n")
                altNames = [x.strip() for x in namesList]
    except TypeError:
        pass


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


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect_image(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    custom_image_bgr = image  # use: detect(,,imagePath,)
    custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    # import scipy.misc
    # custom_image = scipy.misc.imread(image)
    # you should comment line below: free_image(im)
    im, arr = array_to_image(custom_image)

    num = c_int(0)
    if debug:
        print("Assigned num")
    pnum = pointer(num)
    if debug:
        print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    if debug:
        print("did prediction")
    dets = get_network_boxes(
        net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box)  # OpenCV

    # dets = get_network_boxes(net, im.w, im.h, thresh,
    #                          hier_thresh, None, 0, pnum, letter_box)
    if debug:
        print("Got dets")
    num = pnum[0]
    if debug:
        print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug:
        print("did sort")
    res = []
    if debug:
        print("about to range")
    for j in range(num):
        if debug:
            print("Ranging on "+str(j)+" of "+str(num))
        if debug:
            print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug:
                print("Class-ranging on "+str(i)+" of " +
                      str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug:
        print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug:
        print("did sort")
    free_detections(dets, num)
    if debug:
        print("freed detections")
    return res



def performDetect(thresh=0.25,
                  configPath=configPath,
                  weightPath=weightPath,
                  metaPath=metaPath,
                  showImage=True):
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    # if is used cv2.imread(image)
    detect_object_alive = True
    # print('detect_object_alive is', detect_object_alive )
    detections = detect_image(netMain, metaMain, image_0, thresh)
    # detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    if showImage:
        try:
            imcaption = []
            for detection in detections:
                img = image_0
                if detection[0] == 'turn_left_traffic':
                    turn_left = True
                elif detection[0] == 'turn_right_traffic':
                    turn_right = True
                x_center = detection[-1][0]
                y_center = detection[-1][1]
                width = detection[-1][2]
                height = detection[-1][3]
                x_top = int(x_center - width/2)
                y_top = int(y_center - height/2)
                x_bot = int(x_top + width)
                y_bot = int(y_top + height)
                cv2.rectangle(img, (x_top, y_top),
                              (x_bot, y_bot), (255, 255, 255), 2)
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                cv2.imshow('object', img)
        except Exception as e:
            print("Unable to show image: "+str(e))
        cv2.waitKey(1)        
    detect_object_alive = False



"""
:param image_path:
:param weights_path:
:return:
"""
input_tensor = tf.placeholder(dtype=tf.float32, shape=[
                              1, 256, 512, 3], name='input_tensor')

net = lanenet.LaneNet(phase='test', net_flag='vgg')
binary_seg_ret, instance_seg_ret = net.inference(
    input_tensor=input_tensor, name='lanenet_model')
saver = tf.train.Saver()
# Set sess configuration
sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
sess_config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=sess_config)
saver.restore(sess=sess, save_path=weights_path)

def inference_net():
    global detect_lane_alive,image_0
    detect_lane_alive = True
    # preprocess
    image = image_0
    crop_img = image[skyLine:, 0:]
    h = crop_img.shape[0]  # 240
    w = crop_img.shape[1]  # 320

    # resize for lanenet image required
    image = cv2.resize(crop_img, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0

    # get image with lane
    binary_seg_image, instance_seg_image = sess.run(
        [binary_seg_ret, instance_seg_ret],
        feed_dict={input_tensor: [image]}
    )
    # reed image
    embedding_image = np.array(instance_seg_image[0], np.uint8)
    # resize to crop image size
    embedding_image = cv2.resize(
        embedding_image, (w, h), interpolation=cv2.INTER_LINEAR)
    # embedding_image = cv2.resize(embedding_image, (512, 256), interpolation=cv2.INTER_LINEAR)

    # make equal with src image
    top = skyLine
    bottom = left = right = 0
    embedding_image = cv2.copyMakeBorder(
        embedding_image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    # print(embedding_image.shape)
    # draw lines
    h = embedding_image.shape[0]
    w = embedding_image.shape[1]
    left_lane = []
    right_lane = []

    # get line coordinates

    max_y = 120
    dx = car_x
    zero_image = np.zeros((h, w, 3))
    lanes = []
    for y in range(100, 240, 1):
        x = 5
        lane_number = 0
        while x < 315:
            if(embedding_image[y, x].any() > 0):
                # zero_image[y][x] = [0, 255, 0]
                # print('lane point begin', lane_number)
                # find new lane
                if len(lanes) <= lane_number + 1:
                    lanes.append([])
                    # print('create new lane:', lane_number,
                    #       'with coordinates:', 'x:', x, '- y:', y)
                if len(lanes[lane_number]) == 0:
                    lanes[lane_number].append([x, y])
                    # print('find lane:', lane_number,
                    #       'with coordinates:', x, y,
                    #       '\nlanes append:', lanes[lane_number][0])
                    lane_number += 1
                # when x in the old lane
                elif abs(x - lanes[lane_number][-1][0]) < 10:
                    if y - lanes[lane_number][-1][1] < 10:
                        # print(abs(x - lanes[lane_number][-1][0]))
                        lanes[lane_number].append([x, y])
                        # print('find lane:', lane_number,
                        #       'with coordinates:', x, y)
                        lane_number += 1
                #x in the next lane
                elif abs(x - lanes[lane_number][-1][0]) > 50:
                    lane_number += 1
                    # print('x in the next lane:', lane_number,
                    #       'with coordinates:', x, y)
                    lanes[lane_number].append([x, y])
                    lane_number += 1
                # print('lane point end', lane_number)
                # # print image found
                # print('end loop')
                zero_image[y][x] = [255, 0, 0]

                x += 40
            else:
                x += 1
    # y_left_end = y_right_end = 80
    # x_left_end = 0
    # x_right_end = 320

    y_left_top= y_right_top= 180
    x_left_top= 0
    x_right_top= 320
    for lane in lanes:
        if len(lane) > 20:
        # print(lane[-1])
        # find coord end
        # if lane[-1][0] < car_x - 30 and y_left_end < lane[-1][1]:
        #     y_left_end = lane[-1][1]
        #     x_left_end = lane[-1][0]
        # elif lane[-1][0] > car_x + 30 and y_right_end < lane[-1][1]:
        #     y_right_end = lane[-1][1]
        #     x_right_end = lane[-1][0]
        # print(lane[0])
        # find coord top
            if lane[-1][0] < car_x - 30 and y_left_top > lane[0][1]:
                y_left_top= lane[0][1]
                x_left_top= lane[0][0]
            elif lane[-1][0] > car_x + 30 and y_right_top> lane[0][1]:
                y_right_top= lane[0][1]
                x_right_top= lane[0][0]
    print(len(lanes))
    # print('find end', 'left', x_left_end, y_left_end,
    #       'right', y_right_end, x_right_end)
    # print('find top', 'left', y_left_top, x_left_top,
    #       'right', y_right_top, x_right_top)
    # In case of error, don't draw the line

    # if draw_left:
    #     cv2.line(zero_image, (x_left_top,y_left_top),
    #              (x_left_end, y_left_end),  [0, 0, 255], 5)
    # if draw_right:
    #     cv2.line(zero_image, (x_right_top, y_right_top),
    #              (x_right_end, y_right_end), [255, 0, 0], 5)
    dx = int((x_left_top + x_right_top)/2)
    drive_follow_lane(dx)
    # cv2.imwrite(
    #     '/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/test.jpg', zero_image)
    cv2.imshow('test', zero_image)
    cv2.imshow('test2',embedding_image)
    cv2.waitKey(1)
    detect_lane_alive = False




# def get_lane_begin_end(lanes):
#     global y_left_end,y_right_end,x_left_end,x_right_end,y_left_top,y_right_top,x_left_top,x_right_top
#     for lane in lanes:
#         if len(lane) < 10:
#             lanes.pop(lanes.index(lane))
#         # print(lane[-1])
#         # find coord end
#         if lane[-1][0] < car_x - 30 and y_left_end < lane[-1][1]:
#             y_left_end = lane[-1][1]
#             x_left_end = lane[-1][0]
#         elif lane[-1][0] > car_x + 30 and y_right_end < lane[-1][1]:
#             y_right_end = lane[-1][1]
#             x_right_end = lane[-1][0]
#         # print(lane[0])
#         # find coord top
#         if lane[0][0] < car_x - 30 and y_left_top> lane[0][1]:
#             y_left_top= lane[0][1]
#             x_left_top= lane[0][0]
#         elif lane[0][0] > car_x + 30 and y_right_top> lane[0][1]:
#             y_right_top= lane[0][1]
#             x_right_top= lane[0][0]

    # zero_image[y_left_end][x_left_end] = [0, 255, 0]
    # zero_image[y_right_end][x_right_end] = [0, 255, 0]
    # zero_image[y_left_top][x_left_top] = [0, 0, 255]
    # zero_image[y_right_top][x_right_top] = [0, 0, 255]
    # zero_image = np.zeros((240, 320, 3))




# def draw_lines(img, color=[0, 0, 255], thickness=5):
#     global angle

    # # In case of error, don't draw the line
    # draw_right = True
    # draw_left = True

    # if draw_left:
    #     cv2.line(img, (x_left_top,y_left_top),
    #              (x_left_end, y_left_end),  color, thickness)
    # if draw_right:
    #     cv2.line(img, (x_right_top, y_right_top),
    #              (x_right_end, y_right_end), color, thickness)
    # if draw_left and draw_right:
    #     x_des = int((x_left_end + x_right_end)/2)
    #     # y_des = int((y_left_end + y_right_end)/2)
    # drive_follow_lane(x_des)
    # dx = x_des - car_x
    # dy = car_x - y_des
    # if dx < 0:
    #     angle = -np.arctan(-dx/dy) * 180/math.pi
    # elif dx == 0:
    #     angle = 0
    # else:
    #     angle = np.arctan(dx/dy) * 180/math.pi
    # print('angle is:', angle)


def drive_follow_lane(dx):
    global speed, angle
    l = dx - 10
    r = dx + 10
    if(car_x < l):
        # print("Drive to the right")
        if angle < 0:
            angle = 0
        angle += 1
    elif(car_x > r):
        # print("Drive to the left")
        if angle > 0:
            angle = 0
        angle -= 1
    else:
        angle = 0
        # print("go ahead")
    # print(dx, car_x)
    # print('Angle:', angle)

def car_control(angle, speed):
    pub_speed = rospy.Publisher(
        '/bridge05/ithutech/set_speed', Float32, queue_size=10)
    pub_angle = rospy.Publisher(
        '/bridge05/ithutech/set_angle', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle.publish(angle)
    # print('Angle:', angle, 'Speed:', speed)


def image_callback(data):
    global frame, image_0, image_object, have_data, \
        detect_object_alive, detect_lane_alive
    car_control(angle, speed)
    temp = np.fromstring(data.data, np.uint8)
    image_0 = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    frame += 1
    if frame > 5:
        print('found data, check: ', detect_object_alive, detect_lane_alive)
        if not detect_object_alive:
            print('detect object')
            threading.Thread(target=performDetect())
        else:
            print('detect_object is running')
        frame = 0
    if not detect_lane_alive:
        print('detect lane')
        threading.Thread(target=inference_net())
    else:
        print('lane thread is running')
    
    cv2.imshow('object', image_0)
    # cv2.imshow('detect_objetct',img)


avoid_object = False
br = False


def main():
    rospy.init_node('ithutech_node', anonymous=True)
    # rospy.Subscriber("/ithutech_image/compressed", CompressedImage,
    #                  image_callback, queue_size=1, buff_size=2**24)
    rospy.Subscriber("/bridge05/ithutech/camera/rgb/compressed", CompressedImage,
                     image_callback, queue_size=1, buff_size=2**24)
    rospy.spin()


if __name__ == '__main__':
    main()
