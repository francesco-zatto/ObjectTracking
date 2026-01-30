import numpy as np
import cv2
import os

EXPERIMENT_NAME = 'Antoine-Mug'
EXPERIMENT_PATH = os.path.join('../Experiments', EXPERIMENT_NAME)
if not os.path.exists(EXPERIMENT_PATH):
    os.makedirs(EXPERIMENT_PATH)
CHANNELS = "HS"

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
    global r,c,w,h,roi_defined
    # if the left mouse button was clicked, 
    # record the starting ROI coordinates 
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2-r)
        w = abs(c2-c)
        r = min(r,r2)
        c = min(c,c2)  
        roi_defined = True
        
def show_channels_and_weigths(hsv, weights):
    hue = hsv.copy()
    hsv2 = np.full(frame.shape[:2], 255)
    hue[:,:,1] = hsv2
    hue[:,:,2] = hsv2
    hue_rgb = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)
    cv2.imshow("Hue", hue_rgb)

    sat = hsv[:,:,1]
    cv2.imshow("Saturation", sat)
    
    val = hsv[:,:,2]
    cv2.imshow("Value", val)
    
    cv2.imshow("Weights", weights)
    return hue_rgb, sat, val, weights

def normalize_bgr_frame(frame):
    """Given a frame, returns a new one where at each pixel we apply (B,G,R) = (B,G,R) / (B + G + R)"""
    f = frame.astype(np.float32)
    b, g, r = cv2.split(f)

    channel_sum = b + g + r
    channel_sum[channel_sum == 0] = 1e-6 

    r_norm = r / channel_sum
    g_norm = g / channel_sum
    b_norm = b / channel_sum

    norm_frame = cv2.merge([b_norm, g_norm, r_norm])
    return (norm_frame * 255).astype(np.uint8)

def updateModel(alpha: float = 0, new_roi_hist=None):
    global roi_hist
      
    if alpha==0:
        return
    
    roi_hist = cv2.addWeighted(roi_hist, (1 - alpha), new_roi_hist, alpha, 0)

def computeRoiHist(hsv_roi, channels_str="H"):
    channel_indices = []
    channels_hist_sizes = []
    channels_range = []

    if "H" in channels_str:
        channel_indices.append(0)
        channels_hist_sizes.append(180)
        channels_range.extend([0, 180])
    if "S" in channels_str:
        channel_indices.append(1)
        channels_hist_sizes.append(256)
        channels_range.extend([30, 256])
    if "V" in channels_str:
        channel_indices.append(2)
        channels_hist_sizes.append(256)
        channels_range.extend([20, 236])
        
    # Pixels with S<30, V<20 or V>235 are ignored 
    mask = cv2.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
    
    roi_hist = cv2.calcHist([hsv_roi], channel_indices, mask, channels_hist_sizes, channels_range)
    return cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../Sequences/Antoine-Mug.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = computeRoiHist(hsv_roi, channels_str=CHANNELS)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


cpt = 1
while(1):
    ret ,frame = cap.read()
    # frame = normalize_bgr_frame(frame)
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Backproject the model histogram roi_hist onto the 
        # current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        # dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) # hue
        # dst = cv2.calcBackProject([hsv],[1],roi_hist,[0,256],1) # saturation 
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 30, 256], 1)

        hue_rgb, sat, val, weights = show_channels_and_weigths(hsv, dst)

        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        updateModel(0, computeRoiHist(hsv[c:c+w, r:r+h], channels_str=CHANNELS))

        # Draw a blue rectangle on the current image
        r,c,h,w = track_window
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Frame_%04d.png'%cpt),frame_tracked)
            cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Hue_%04d.png'%cpt),hue_rgb)
            cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Sat_%04d.png'%cpt),sat)
            cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Val_%04d.png'%cpt),val)
            cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Weights_%04d.png'%cpt),weights)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()