import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

import sys 


# capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
def build_filters():
    filters = []
    ksize = 10
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

# loop runs if capturing has been initialized 
while(1): 
    ret, im = cap.read()
    scale_percent = 60 # percent of original size
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	#gabor_face = cv2.getGaborKernel(ksize=(5,5), sigma=, theta, lambda, gamma, psi, ktype)
    
    cv2.imshow('Original', im)
    cv2.imshow('Gray', gray)
    #cv2.imshow(,  gabor_face)
    
    filters = build_filters()
    
    gabor_on_gray = process(gray, filters)
    gabor_on_color = process(im, filters)
    
    cv2.imshow('Gabor on Color', gabor_on_color)
    cv2.imshow('Gabor on Gray', gabor_on_gray)
    
    print("Details of Original :")
    print(" Max,Min=",np.max(im),np.min(im))
    print(" Shape=", im.shape)
    
    print("Details of Gray :")
    print(" Max,Min=",np.max(gray),np.min(gray))
    print(" Shape=", gray.shape)
    
    print("Details of Gabor on Color :")
    print(" Max,Min=",np.max(gabor_on_color),np.min(gabor_on_color))
    print(" Shape=", gabor_on_color.shape)
    
    print("Details of Gabor on Gray :")
    print(" Max,Min=",np.max(gabor_on_gray),np.min(gabor_on_gray))
    print(" Shape=", gabor_on_gray.shape)
    
    k = cv2.waitKey(5) & (0xFF == ord('q')) # cv2.waitKey(1) & 0xFF == ord('q'):
    if k == True: 
        break


# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  