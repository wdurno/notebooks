## this is basically just ball_tracker.py (linked), reduced to an API 
## link: https://github.com/sunfounder/SunFounder_PiCar-V/blob/V3.0/ball_track/ball_tracker.py 
## 
## primary interface: 
## - img ## img.read(), img.release() 
## - find_blob() ## returns (x, y), radius 
## - fw ## front wheels, ie. fw.turn(90)  
## - bw ## back wheels, ie. bw.speed = 60, bw.stop(), bw.forward() 


from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os
import imutils 
from .constants import SCREEN_WIDTH, SCREEN_HEIGHT, CENTER_X, CENTER_Y, BALL_SIZE_MIN, BALL_SIZE_MAX 

picar.setup()
# Show image captured by camera, True to turn on, you will need #DISPLAY and it also slows the speed of tracking
show_image_enable   = False
draw_circle_enable  = False
scan_enable         = False
rear_wheels_enable  = True
front_wheels_enable = True
pan_tilt_enable     = True

if (show_image_enable or draw_circle_enable) and "DISPLAY" not in os.environ:
    print('Warning: Display not found, turn off "show_image_enable" and "draw_circle_enable"')
    show_image_enable   = False
    draw_circle_enable  = False

kernel = np.ones((5,5),np.uint8)
img = cv2.VideoCapture(-1)

if not img.isOpened:
    print("not open")
else:
    print("open")
    
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HEIGHT)

# Filter setting, DONOT CHANGE
hmn = 12
hmx = 37
smn = 96
smx = 255
vmn = 186
vmx = 255

CAMERA_STEP = 2
CAMERA_X_ANGLE = 20
CAMERA_Y_ANGLE = 20

MIDDLE_TOLERANT = 5
PAN_ANGLE_MAX   = 170
PAN_ANGLE_MIN   = 10
TILT_ANGLE_MAX  = 150
TILT_ANGLE_MIN  = 70
FW_ANGLE_MAX    = 90+30
FW_ANGLE_MIN    = 90-30

SCAN_POS = [[20, TILT_ANGLE_MIN], [50, TILT_ANGLE_MIN], [90, TILT_ANGLE_MIN], [130, TILT_ANGLE_MIN], [160, TILT_ANGLE_MIN],
            [160, 80], [130, 80], [90, 80], [50, 80], [20, 80]]

bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
picar.setup()

fw.offset = 0
pan_servo.offset = 10
tilt_servo.offset = 0

bw.speed = 0
fw.turn(90)
pan_servo.write(90)
tilt_servo.write(90)

motor_speed = 60

def nothing(x):
    pass

def destroy():
    bw.stop()
    img.release()

def test():
    fw.turn(90)

def find_blob(bgr_image) :
    radius = 0
    # Load input image
    #_, bgr_image = img.read()
    ##ret, bgr_image = img.read()
    ##if ret == False:
    ##    print("Failed to read image")

    orig_image = bgr_image

    bgr_image = cv2.medianBlur(bgr_image, 3)

    # Convert input image to HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image, keep only the red pixels
    lower_red_hue_range = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    upper_red_hue_range = cv2.inRange(hsv_image, (160, 100, 100), (179, 255, 255))
    # Combine the above two images
    red_hue_image = cv2.addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0)

    red_hue_image = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)

    # Use the Hough transform to detect circles in the combined threshold image
    circles = cv2.HoughCircles(red_hue_image, cv2.HOUGH_GRADIENT, 1, 120, 100, 20, 10, 0)

    if circles is not None:
        circles = np.uint16(np.around(circles))

    # Loop over all detected circles and outline them on the original image
        all_r = np.array([])
    # print("circles: %s"%circles)
        try:
            for i in circles[0,:]:
                # print("i: %s"%i)
                all_r = np.append(all_r, int(round(i[2])))
            closest_ball = all_r.argmax()
            center=(int(round(circles[0][closest_ball][0])), int(round(circles[0][closest_ball][1])))
            radius=int(round(circles[0][closest_ball][2]))
            if draw_circle_enable:
                cv2.circle(orig_image, center, radius, (0, 255, 0), 5)
        except IndexError:
            pass
            #print("circles: %s"%circles)

    # Show images
    if show_image_enable:
        cv2.namedWindow("Threshold lower image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Threshold lower image", lower_red_hue_range)
        cv2.namedWindow("Threshold upper image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Threshold upper image", upper_red_hue_range)
        cv2.namedWindow("Combined threshold images", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Combined threshold images", red_hue_image)
        cv2.namedWindow("Detected red circles on the input image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Detected red circles on the input image", orig_image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        return (0, 0), 0
    if radius > 3:
        return center, radius
    else:
        return (0, 0), 0 

def find_blob_contours(rgb_image):
    red_hsv_lower = (90, 90, 90)
    red_hsv_upper = (200, 200, 255)
    image = cv2.GaussianBlur(rgb_image, (9, 9), 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = cv2.inRange(image, red_hsv_lower, red_hsv_upper)
    image = cv2.erode(image, None, iterations=2)
    image = cv2.dilate(image, None, iterations=2)
    contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > 3:
            return (x, y), r
        pass
    return (0, 0), 0
