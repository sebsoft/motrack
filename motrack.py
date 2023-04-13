#!/usr/bin/python3
from __future__ import print_function
PROG_VERSION = "1.51"

import logging
# Setup Logging


print("Loading Python Libraries ...")
import os
import sys
import time
import datetime
import math
import cv2
import subprocess
import logging
from timeit import default_timer as timer
import user_code_speed

CONFIG_FILENAME = "config.py"  # Settings variables file to impor


if os.path.exists(CONFIG_FILENAME):
    # Read Configuration variables from config.py file
    try:
        #logging.info("Import Settings from %s", CONFIG_FILENAME)
        from config import *
    except ImportError:
        #logging.error("Problem Importing configuration variables from %s" %
        #              CONFIG_FILENAME)
        sys.exit(1)
else:
    #logging.error("Configuration File Not Found %s" % CONFIG_FILENAME)
    sys.exit(1)


if loggingToFile:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-8s %(funcName)-10s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=logFilePath,
            filemode="a",
            )
    





PROG_NAME = os.path.basename(__file__)

# ------------------------------------------------------------------------------
def validate_cam(cam, camlist):
    ''' Make sure camera value is correct (case insensitive)
    '''
    camlow = cam.lower()
    if not camlow in camlist:
        logging.error('%s Not a Valid Camera Value', cam)
        logging.info('Valid Values are %s', ' '.join(camlist))
        logging.info('Edit config.py CAMERA variable.')
        sys.exit(1)
    else:
        logging.info('Connecting to camera %s', camlow)
    return camlow


# ------------------------------------------------------------------------------
def show_settings(filename):
    '''
    Display program configuration variable settings
    read config file and print each decoded line
    '''
    with open(filename, 'rb') as f:
        for line in f:
            print(line.decode().strip())
    print("")


# ------------------------------------------------------------------------------
def get_image_name(path, prefix):
    '''
    build image file names by number sequence or
    date/time Added tenth of second
    '''
    rightNow = datetime.datetime.now()
    filename = "%s/%s%04d%02d%02d-%02d%02d%02d%d.jpg" % (
        path,
        prefix,
        rightNow.year,
        rightNow.month,
        rightNow.day,
        rightNow.hour,
        rightNow.minute,
        rightNow.second,
        rightNow.microsecond / 100000,
    )
    return filename


# ------------------------------------------------------------------------------
def timer_end(timer_start, timer_sec):
    '''
    Check if timer has expired
    Return updated start time status of expired timer True or False
    '''
    rightNow = datetime.datetime.now()
    timeDiff = (rightNow - timer_start).total_seconds()
    if timeDiff >= timer_sec:
        return True
    else:
        return False


# ------------------------------------------------------------------------------
def get_motion_track_point(grayimage1, grayimage2):
    '''
    Process two grayscale images.
    check for motion and return center point
    of motion for largest contour.
    '''
    movement_center = None
    movement_size = 0
    # Get differences between the two greyed images
    differenceimage = cv2.absdiff(grayimage1, grayimage2)
    # Blur difference image to enhance motion vectors
    differenceimage = cv2.blur(differenceimage, (BLUR_SIZE, BLUR_SIZE))
    # Get threshold of blurred difference image
    # based on THRESHOLD_SENSITIVITY variable
    retval, thresholdimage = cv2.threshold(
        differenceimage, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY
    )
    try:
        # opencv2 syntax default
        contours, hierarchy = cv2.findContours(
            thresholdimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    except ValueError:
        # opencv 3 syntax
        thresholdimage, contours, hierarchy = cv2.findContours(
            thresholdimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    if contours:
        c = max(contours, key = cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        movement_size = w * h
        if movement_size <= TRACK_MIN_AREA:
            return None, movement_size
        movement_center = (int(x + w / 2) + x_left, int(y + h / 2) + y_upper)
        movement_size = w * h
    return movement_center, movement_size


# ------------------------------------------------------------------------------
def track_motion_direction(xy1, xy2):
    '''
    Return the direction of travel between tracking locations
    '''
    x1, y1 = xy1
    x2, y2 = xy2
    
    trackdirection = "NONE"
    dist = x1 - x2

    if dist < 0 :
        trackdirection = "L2R"
    elif dist > 0 :
        trackdirection = "R2L"
    
    logging.info("Distance %i direction %s", dist, trackdirection)
    return trackdirection


# ------------------------------------------------------------------------------
def track_motion_distance(xy1, xy2):
    '''
    Return the triangulated distance between two tracking locations
    '''
    x1, y1 = xy1
    x2, y2 = xy2
    trackLen = int(abs(math.hypot(x2 - x1, y2 - y1)))
    return trackLen


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if SHOW_SETTINGS:
        show_settings(CONFIG_FILENAME)

    if not os.path.exists(IM_DIR):  # Check if image directory exists
        os.makedirs(IM_DIR)  # Create directory if Not Found

    

    logging.info("%s ver %s written by Claude Pageau", PROG_NAME, PROG_VERSION)
    
    
    
    mycam = "pilibcam"
    if mycam is not None:
        logging.info("Start Stream Thread: %s", mycam)
        if mycam == "pilibcam":
            if not os.path.exists('streampilibcam.py'):
                logging.error("streampilibcam.py File Not Found.")
                sys.exit(1)
            try:
                from streampilibcam import PiLibCamStream
            except ImportError:
                logging.error("Failed Import of streampilibcam.py")
                sys.exit(1)                    
            vs = PiLibCamStream(size=IM_SIZE,
                                im_vflip=IM_VFLIP,
                                im_hflip=IM_HFLIP).start()

        logging.info("Wait ...")
        time.sleep(5)  # Allow Camera to warm up        
    # initialize first gray image
    
    start_track = True
    track_hist = []
    image1_full = vs.read()
    image1 = image1_full[y_upper:y_lower, x_left:x_right]
    grayimage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    im_height, im_width, _ = image1_full.shape
    if TRACK_TRIG_AUTO:
        # Auto calculate variables below
        TRACK_TRIG_LEN = int(im_width / 8)  # auto calc track len.
        TRACK_INTERVAL_LEN = int(TRACK_TRIG_LEN / 2.0) # Max allowed px distance from previous track point
        logging.info("Auto Calculated TRACK_TRIG_LEN=%i and TRACK_INTERVAL_LEN=%i",
                      TRACK_TRIG_LEN, TRACK_INTERVAL_LEN)
    logging.info("Start %s Stream Thread" % mycam.upper())
    logging.info("Start Motion Tracking Loop. Ctrl-c Quits ...")
    logging.info("--------------------------------------------")
    tracking = True
    try:
        while tracking:
            image2_full = vs.read()
            image2 = image2_full[y_upper:y_lower, x_left:x_right]
            try:
                grayimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            except:
                continue

            motion_xy, motion_size = get_motion_track_point(grayimage1, grayimage2)
            grayimage1 = grayimage2  # update for next track

            # No motion try again
            if motion_xy is None:
                continue

            # Check if timer expired and if so restart track and try again
            if not start_track and timer_end(track_timer_start, TRACK_TIMEOUT_SEC):
                if LOGGING_ON:
                    logging.info("TIMER: GT %i TRACK_TIMEOUT_SEC",
                                 TRACK_TIMEOUT_SEC)
                start_track = True
                continue
            # If this is first track motion initialize tracking variables
            if motion_xy and start_track:
                
                track_start  = timer()
                track_timer_start = datetime.datetime.now()
                image1_full = image2_full
                filename1  = get_image_name(IM_DIR,"")
                
                mpoint_start = motion_xy
                prev_mpoint = motion_xy
                max_radius = TRACK_INTERVAL_LEN
                if LOGGING_ON:
                    logging.info("START: (%i, %i) %i sqpx",
                                 mpoint_start[0], mpoint_start[1], motion_size)
                if TRACK_HIST_ON: # Reset Tracking History
                    track_hist = [mpoint_start]
                start_track = False
            # Start tracking motion
            else:
                
                mpoint2 = motion_xy
                track_length = track_motion_distance(mpoint_start, mpoint2)
                
               
                if TRACK_HIST_ON:
                    track_hist.append(mpoint2)  # Append current mpoint to track history
                if GUI_ON and CIRCLE_ON:
                    cv2.circle(image2_full,
                               mpoint2,
                               CIRCLE_SIZE,
                               LINE_COLOR,
                               LINE_THICKNESS)
                if track_length <= max_radius:
                    max_radius = track_length + TRACK_INTERVAL_LEN
                else:
                    # ignore out of range points and reset start point
                    mpoint_start = mpoint2
                    if LOGGING_ON:
                        logging.info("(%i, %i) Track Reset: Radius %i Exceeds %i",
                        mpoint2[0], mpoint2[1], track_length, max_radius)
                    start_track = True
                    continue
                if track_length > TRACK_TRIG_LEN:
                    # This was a valid track
                    track_direction = track_motion_direction(mpoint_start, mpoint2)
                    track_stop  = timer()
                    tracktime = (track_stop - track_start)*1000

                    speed = round(((track_length * PIXELS_METER_R2L ) / (tracktime/1000)) * 3.6 , 0) 
                    if LOGGING_ON:
                        logging.info("Track End: (%i, %i) Length %i GT %i TRACK_TRIG_LEN px in %i ms %s speed %i",
                                     mpoint2[0], mpoint2[1], track_length, TRACK_TRIG_LEN,tracktime, track_direction, speed)
                    filename2 = get_image_name(IM_DIR, "")
                    
                    
                    # Resize image before saving ..
                    if CIRCLE_ON:  # Put Circle on last motion xy before saving image
                        if TRACK_HIST_ON:
                            for point in track_hist:
                                cv2.circle(image2_full,
                                           point,
                                           CIRCLE_SIZE,
                                           LINE_COLOR,
                                           LINE_THICKNESS)
                        else:
                            cv2.circle(image2_full,
                                       mpoint2,
                                       CIRCLE_SIZE,
                                       LINE_COLOR,
                                       LINE_THICKNESS)
                    
                    #im_resized = cv2.resize(image2_full, (int(im_width * IM_BIGGER),
                    #                             int(im_height * IM_BIGGER)))
                    
                    if track_direction == "R2L":
                        logging.info("SAVE R2L!: %s", filename1)
                        logging.info("SAVE R2L!: %s", filename2)
                        cv2.imwrite(filename1, image1_full)
                        cv2.imwrite(filename2, image2_full)
                        user_code_speed.userMotionCode(filename1,filename2,tracktime,speed
                                        )
                        
                    
                    time.sleep(TRACK_DELAY_SEC)  # Delay before starting another track
                    start_track = True
                #else:
                    #if LOGGING_ON:
                        #logging.info("(%i, %i)", mpoint2[0], mpoint2[1])

            if not start_track and timer_end(track_timer_start, TRACK_TIMEOUT_SEC):
                if LOGGING_ON:
                    logging.info("Track Timeout: GT %i TRACK_TIMEOUT_SEC",
                                 TRACK_TIMEOUT_SEC)
                start_track = True

            if GUI_ON:
                cv2.imshow("MoTrack (q in Window Quits)", image2)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()


    except KeyboardInterrupt:
        print("")
        logging.info("User Pressed Keyboard ctrl-c")
        logging.info("Exiting %s ver %s", PROG_NAME, PROG_VERSION)
        logging.info("Stop Stream Thread: %s", cam)
        logging.info("Wait ...")
        vs.stop()
        sys.exit(0)
