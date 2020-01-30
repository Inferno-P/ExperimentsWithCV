import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

import sys 

im = cv2.imread('scene.png')

# capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
  
# loop runs if capturing has been initialized 
while(1): 
    ret, im = cap.read() 
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    output = im.copy()
    
    for c in cnts:
        # draw each contour on the output image with a 3px thick purple
        # outline, then display the output contours one at a time
        cv2.drawContours(output, [c], -1, (240, 0, 159), 1)
        cv2.imshow("Contours", output)
    
    
    k = cv2.waitKey(5) & (0xFF == ord('q')) # cv2.waitKey(1) & 0xFF == ord('q'):
    if k == 27: 
        break
  
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  