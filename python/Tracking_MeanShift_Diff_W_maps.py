import numpy as np
import cv2
import os

EXPERIMENT_NAME = 'SunShade'
EXPERIMENT_PATH = os.path.join('../Frames', EXPERIMENT_NAME)
if not os.path.exists(EXPERIMENT_PATH):
    os.makedirs(EXPERIMENT_PATH)

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
    global r,c,w,h,roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h, w = abs(r2-r), abs(c2-c)
        r = min(r,r2)
        c = min(c,c2)  
        roi_defined = True

def get_edge_weight_map(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize to [0,255]
    edge_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return edge_map.astype(np.uint8)


def show_channels_and_weigths(hsv, weights=None):
    hue = hsv.copy()
    hsv2 = np.full(hsv.shape[:2], 255, dtype=np.uint8)
    hue[:,:,1] = hsv2
    hue[:,:,2] = hsv2
    hue_rgb = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)
    cv2.imshow("Hue", hue_rgb)
    cv2.imshow("Saturation", hsv[:,:,1])
    cv2.imshow("Value", hsv[:,:,2])
    if weights is not None:
        cv2.imshow("Weights - Combined", weights)
    return hue_rgb, hsv[:,:,1], hsv[:,:,2], weights

def computeRoiHist(hsv_roi, channels_str="H"):
    if channels_str == "H":
        hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    elif channels_str == "V":
        hist = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
    return cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)

# --- STARTING VIDEO ---
cap = cv2.VideoCapture(f'../Sequences/VOT-Basket.mp4')
ret, frame = cap.read()
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
while True:
    cv2.imshow("First image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or roi_defined: break

track_window = (r, c, h, w)
roi = frame[c:c+w, r:r+h]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 1. Compute Base Histograms
roi_hist_hue = computeRoiHist(hsv_roi, "H")

# 2. Determine BGR weights based on Hue Direction
avg_h = np.mean(hsv_roi[:,:,0])
if 100 < avg_h < 130:   bgr_weights = (0.1, 0.45, 0.45) # Blue target -> de-emphasize Blue
elif 40 < avg_h < 80:   bgr_weights = (0.45, 0.1, 0.45) # Green target -> de-emphasize Green
elif avg_h < 20 or avg_h > 160: bgr_weights = (0.45, 0.45, 0.1) # Red target -> de-emphasize Red
else: bgr_weights = (0.33, 0.33, 0.33)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)
cpt = 1

while True:
    ret, frame = cap.read()
    if not ret: break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Generate Weight Maps
    dst_hue = cv2.calcBackProject([hsv], [0], roi_hist_hue, [0, 180], 1)
    dst_edge = get_edge_weight_map(frame)

    dst_combined = cv2.multiply(dst_hue, dst_edge, scale=1/255.0) 

    # Display results
    show_channels_and_weigths(hsv, dst_combined)
    cv2.imshow("Weights - Edge Map", dst_edge)
    cv2.imshow("Weights - Hue Map", dst_hue)

    # MeanShift
    ret, track_window = cv2.meanShift(dst_combined, track_window, term_crit)

    # Draw result
    x, y, ww, hh = track_window
    cv2.rectangle(frame, (x, y), (x+ww, y+hh), (255, 0, 0), 2)
    cv2.imshow('Sequence', frame)

    k = cv2.waitKey(60) & 0xFF
    if k == 27: break
    if k == ord('s'):
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Frame_%04d.png'%cpt), frame)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Weights_Edge_%04d.png'%cpt), dst_edge)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Weights_Hue_%04d.png'%cpt), dst_hue)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Weights_Combined_%04d.png'%cpt), dst_combined)
    cpt += 1

cap.release()
cv2.destroyAllWindows()