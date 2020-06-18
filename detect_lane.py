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

speed = 20
angle = 0
car_x = 160
skyLine = 80
srcPath = '/home/luong/luong_ws/py_digitalrace_2019/src/ithutech/src/'
weights_path = srcPath + 'model/tusimple_lanenet_vgg.ckpt'
image_path = '/home/luong/test.jpg'
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


def performDetect(image,
                  thresh=0.25,
                  configPath=configPath,
                  weightPath=weightPath,
                  metaPath=metaPath,
                  showImage=True):
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    import cv2
    # if is used cv2.imread(image)
    detections = detect_image(netMain, metaMain, image, thresh)
    # detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    if showImage:
        try:
            imcaption = []
            for detection in detections:
                img = image
                x_center = detection[-1][0]
                y_center = detection[-1][1]
                width = detection[-1][2]
                height = detection[-1][3]
                x_top = int(x_center - width/2)
                y_top = int(y_center - height/2)
                x_bot = int(x_top + width)
                y_bot = int(y_top + height)
                cv2.rectangle(img, (x_top, y_top),
                              (x_bot, y_bot), (0, 255, 0), 2)
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
            # detections = {
            #     "detections": detections,
            #     "image": image,
            #     "caption": "\n<br/>".join(imcaption)
            # }
        except Exception as e:
            print("Unable to show image: "+str(e))
    return detections