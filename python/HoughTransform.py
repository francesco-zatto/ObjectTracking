import numpy as np
import cv2
 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../Sequences/VOT-Ball.mp4')

def compute_gradient_maps(image):
    img_float = image.astype(np.float32) / 255.0
    
    # derivatives using sobel kernel
    Ix = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    if len(image.shape) == 3:
        # magnitude for every channel
        mag_all = np.sqrt(Ix**2 + Iy**2)
        
        # find which channel has the strongest gradient at each pixel
        k_indices = np.argmax(mag_all, axis=2)
        
        # extract the Ix, Iy of the dominant channel
        rows, cols, _ = image.shape
        grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # select Ix, Iy corresponding to the max magnitude channel
        Ix = Ix[grid_y, grid_x, k_indices]
        Iy = Iy[grid_y, grid_x, k_indices]
    
    #  Gradient magnitude
    gradient_magnitude = np.hypot(Ix, Iy)
    
    # Gradient argument 
    gradient_orientation = np.arctan2(-Iy, Ix)

    # Masked gradient orientation
    norm_orientation = (gradient_orientation + np.pi) / (2 * np.pi) * 255
    masked_gradient_orientation = np.zeros_like(image)

    masked_gradient_orientation[:,:,0] = norm_orientation
    masked_gradient_orientation[:,:,1] = norm_orientation
    masked_gradient_orientation[:,:,2] = norm_orientation
    
    mask = gradient_magnitude < 0.25

    masked_gradient_orientation[mask, :] = 0
    masked_gradient_orientation[mask, 2] = 255

    return gradient_magnitude, gradient_orientation, masked_gradient_orientation

def calculate_R_table(image, mag, ori, mask):
    w, h, _ = image.shape
    xc, yc = w/2, h/2

    for i in range(1, h):
        for j in range(1, w):
            p = image[i, j, :]


cpt = 1

while(1):
    ret ,frame = cap.read()
    if ret == True:

        cv2.imshow('Frame', frame)
        mag, ori, m_ori = compute_gradient_maps(frame)

        # Map radians [-pi, pi] to [0, 255] for display
        ori_view = ((ori + np.pi) * (255 / (2 * np.pi))).astype(np.uint8)
        #masked_ori_view = ((m_ori + np.pi) * (255 / (2 * np.pi))).astype(np.uint8)
        cv2.imshow('Orientation', ori_view)
        cv2.imshow('Magnitude',mag)
        cv2.imshow('Masked orientation',m_ori)
        
        cv2.waitKey(60)
        cpt += 1
cv2.destroyAllWindows()
cap.release()
