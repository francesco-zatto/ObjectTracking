import numpy as np
import cv2
from scipy.signal import convolve2d
 
#cap = cv2.VideoCapture(0)

# fix path to relative
cap = cv2.VideoCapture('../Sequences/VOT-Ball.mp4')

def SobelDetector(I, s):
  """ Array*double -> Array*Array*Array """
  Sx = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ], dtype=float)

  Sy = np.array([
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]
  ], dtype=float)

  Ix = np.zeros_like(I)
  Iy = np.zeros_like(I)

  for i in range(3):
      Ix[:,:,i] = convolve2d(I[:,:,i], Sx, mode='same')
      Iy[:,:,i] = convolve2d(I[:,:,i], Sy, mode='same')
  G = np.sqrt(Ix**2+Iy**2)
  Ig = (G >= s) * 255
  return Ix, Iy, Ig


def orientation(Ix, Iy, Ig):
    """ Array[n,m]**3 -> Array[n,m]
        Returns an image of orientation.
    """
    n, m = Ix.shape
    x = np.arange(4)*np.pi/4
    ori = np.stack((np.cos(x), np.sin(x)), axis=1)
    O = np.zeros(Ix.shape)
    for i in range(n):
        for j in range(m):
            if Ig[i, j] > 0:
                v = np.array([Ix[i, j], -Iy[i, j]])/Ig[i, j]
                if Iy[i, j] > 0: v = -v
                prod = np.matmul(ori, v)
                maxi = prod.max()
                imax = np.nonzero(prod == maxi)
                O[i, j] = imax[0][0]+1
    return O

cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:

        cv2.imshow('Frame',frame)
        Ix, Iy, Ig = SobelDetector(frame, 120)
        

        ori = np.zeros_like(frame)

        for i in range(3):
            ori[:,:,i] = orientation(Ix[:,:,i], Iy[:,:,i], Ig[:,:,i])

        #cv2.imshow('Orientation',ori)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()