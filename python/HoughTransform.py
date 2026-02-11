import numpy as np
import cv2
import os

EXPERIMENT_NAME = 'HoughTransformScreenshots'
EXPERIMENT_PATH = os.path.join('../Experiments', EXPERIMENT_NAME)
if not os.path.exists(EXPERIMENT_PATH):
    os.makedirs(EXPERIMENT_PATH)

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

def compute_gradient_maps(image):
    img_float = image.astype(np.float32) / 255.0
    
    # derivatives using sobel kernel
    Ix = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    if image.ndim == 3:
        mag_all = np.hypot(Ix, Iy)
        
        k_indices = np.argmax(mag_all, axis=2)

        Ix = np.take_along_axis(Ix, k_indices[:, :, np.newaxis], axis=2).squeeze()
        Iy = np.take_along_axis(Iy, k_indices[:, :, np.newaxis], axis=2).squeeze()   

    gradient_magnitude = np.hypot(Ix, Iy)
    gradient_orientation = np.arctan2(-Iy, Ix)

    h, w = image.shape[:2]
    masked_gradient_orientation = np.zeros((h, w, 3), dtype=np.uint8) # 3 channels

    norm_orientation = ((gradient_orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

    masked_gradient_orientation[:,:,0] = norm_orientation
    masked_gradient_orientation[:,:,1] = norm_orientation
    masked_gradient_orientation[:,:,2] = norm_orientation
    
    mask = gradient_magnitude < 0.25

    masked_gradient_orientation[mask] = [0, 0, 255]

    return gradient_magnitude, gradient_orientation, masked_gradient_orientation

N_BINS = 8
COMPRESSION_FACTOR = 256//N_BINS

def get_R_table(roi_masked_orientation):
    rows, cols, _ = roi_masked_orientation.shape
    row_c, col_c = rows//2, cols//2
    R_table = [[] for _ in range(N_BINS)]

    mask = roi_masked_orientation[:, :, 2] != 255
    y_coords, x_coords = np.where(mask)

    bins = (roi_masked_orientation[mask, 0] / COMPRESSION_FACTOR).astype(int)
    bins = np.clip(bins, 0, N_BINS - 1)

    dr = y_coords - row_c
    dc = x_coords - col_c

    R_table = []
    for b in range(N_BINS):
        idx = (bins == b)
        R_table.append(np.stack((dr[idx], dc[idx]), axis=1)) # Store as (dr, dc) pairs
        
    return R_table


def houghTransform(masked_orientation, R_table):
    rows, cols, _ = masked_orientation.shape

    counter_image = np.zeros((rows,cols), dtype=int)

    mask = masked_orientation[:, :, 2] != 255
    y_idx, x_idx = np.where(mask)

    bins = (masked_orientation[mask, 0] / COMPRESSION_FACTOR).astype(int)
    bins = np.clip(bins, 0, N_BINS - 1)
    
    for b in range(N_BINS):
            pixel_indices = (bins == b)
            curr_y = y_idx[pixel_indices]
            curr_x = x_idx[pixel_indices]
            
            vectors = R_table[b] # shape (N, 2)
            
            for dr, dc in vectors:
                target_y = curr_y + dr
                target_x = curr_x + dc
                
                valid = (target_y >= 0) & (target_y < rows) & (target_x >= 0) & (target_x < cols)
                
                np.add.at(counter_image, (target_y[valid], target_x[valid]), 1)

    res = np.unravel_index(np.argmax(counter_image), counter_image.shape) # get the peak
    return res, counter_image

cap = cv2.VideoCapture('../Sequences/Antoine_mug.mp4')

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
        cv2.destroyWindow("First image")
        break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]

cpt = 1

mag, ori, m_ori = compute_gradient_maps(frame)
R_table = get_R_table(m_ori[c:c+w, r:r+h])

while(1):
    ret ,frame = cap.read()

    if ret == False:
        break

    # cv2.imshow('Frame', frame)
    mag, ori, m_ori = compute_gradient_maps(frame)

    (row_center, col_center), response = houghTransform(m_ori, R_table)
    
    # top-left for the rectangle (x, y)
    draw_x = col_center - (h // 2)
    draw_y = row_center - (w // 2)

    tracking = frame.copy()
    cv2.rectangle(tracking, (draw_x, draw_y), (draw_x + h, draw_y + w), (255, 0, 0), 2)
    cv2.imshow('Tracking', tracking)
    
    # gradient magnitude and orientation

    # Map radians [-pi, pi] to [0, 255] for display
    ori_view = ((ori + np.pi) * (255 / (2 * np.pi))).astype(np.uint8)
    masked_ori_view = ((m_ori + np.pi) * (255 / (2 * np.pi))).astype(np.uint8)
    mag_view = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imshow('Orientation', ori_view)
    # cv2.imshow('Magnitude',mag_view)
    # cv2.imshow('Masked orientation',m_ori)

    # response of the hough transform

    response_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Response', response_norm)
    
    k = cv2.waitKey(60) & 0xff  
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Frame_%04d.png'%cpt),frame)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Tracking_%04d.png'%cpt),tracking)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'ResponseNorm_%04d.png'%cpt),response_norm)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Orientation_%04d.png'%cpt),ori_view)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'Magnitude_%04d.png'%cpt),mag_view)
        cv2.imwrite(os.path.join(EXPERIMENT_PATH, 'MaskedOrientation_%04d.png'%cpt),m_ori)

    cpt += 1

cv2.destroyAllWindows()
cap.release()
