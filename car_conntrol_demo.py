from ctypes import *
from math import *

carPos = POS(120, 300)
turn_left = False
turn_right = False
go_ahead = False
angle = 0
speed = 0
object_detected = 0


class POS:
    def __init__(self, x, y):
        self.x = x
        self.y = y


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


def get_lane(img):
    lx = 1
    ly = 2
    rx = 1
    ry = 2
    left_lane_pos = POS(lx, ly)
    right_lane_pos = POS(rx, ry)
    left_lane = []
    right_lane = []
    left_lane.append(left_lane)
    right_lane.append(right_lane_pos)
    return left_lane, right_lane


def detect_object(img):
    global object_detected
    detections = []
    object_pos = POS(1, 2)
    for detection in detections:
        if(object_pos.y >= 100):
            if(object_pos.x > carPos.x):
                object_detected = 1
            else:
                object_detected = -1
        return detection
    object_detected = 0


def drive_car(img):
    global angle
    left_lane, right_lane = get_lane(img)

    dy = carPos.y
    for i in left_lane:
        for j in right_lane:
            if(i.y == j.y):
                xmax = (i.x + j.x)/2 + 20
                xmin = (i.x + j.x)/2 - 20
    if(object_detected == 0):
        if(carPos.x > xmax):
            angle -= 1
        elif(carPos.x < xmin):
            angle += 1
        else:
            angle = 0


# run(image):
#     left_lane, right_lane = get_lane(image)
#     go_ahead(left_lane,right_lane)
