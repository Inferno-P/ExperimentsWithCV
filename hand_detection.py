# Downlaod the files from :protoFile = https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt and 
# weightsFile= http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
# and download it into the 'hand' folder.
 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os
import sys
import itertools
import time

def build_filters():
    filters = []
    ksize = 31
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

def live_video(camera_port=0):
        """
        Opens a window with live video.
        :param camera:
        :return:
        """

        protoFile = "hand/pose_deploy.prototxt"
        
        weightsFile = "hand/pose_iter_102000.caffemodel"
        
        nPoints = 22
        
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        
        video_capture = cv2.VideoCapture(camera_port)
        
        inWidth,inHeight = 250,250
        
        

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            print(np.min(frame), np.max(frame))
            
            
            filters = build_filters()

            #gabor_on_color = process(frame, filters)    

            frame = cv2.resize(frame, (inWidth,inHeight), interpolation = cv2.INTER_AREA)
            
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / np.mean(frame), (inWidth, inHeight),                          (0,0,0), swapRB=False, crop=False)
            
            print('inpBlob = ', type(inpBlob), inpBlob.shape)
            
            net.setInput(inpBlob)

            output = net.forward()
            
            points = []
 
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                print("Prob Map = ", probMap.shape)

                #frameHeight, frameWidth = inWidth,inHeight

                probMap = cv2.resize(probMap, (inWidth,inHeight))

                cv2.imshow('Probability Map', probMap)
             
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                threshold =0.1
             
                if prob > threshold :
                    cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 255), thickness=1, lineType=cv2.FILLED)

                    cv2.putText(frame, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1, lineType=cv2.LINE_AA)
             
                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                    
                else :

                    points.append(None)

                print("Points  = ", len(points),'\n', points)
                #cv2.imshow('Output-Keypoints', frame)
                
                POSE_PAIRS = points
                
                
                netFrame = frame.copy() 
                if len(POSE_PAIRS) > 2:
                    
                    for i in range(0,len(POSE_PAIRS)-1):
                        partA = POSE_PAIRS[i]
                        partB = POSE_PAIRS[i+1]
                     
                        if partA and partB:
                            cv2.line(frame, partA, partB, (0, 255, 255), 1)
                            cv2.circle(frame, partA, 1, (0, 0, 255), thickness=1, lineType=cv2.FILLED)
                    
                    jumbled = list(itertools.combinations(POSE_PAIRS, 2))
                    print("Length of Points  = ",  len(POSE_PAIRS), "\nLength of Jumbled = ", len(jumbled))
                    
                    for i in range(0,len( jumbled) -1):
                        partA = jumbled[i][0]
                        partB = jumbled[i][1]
                     
                        if partA and partB:
                            cv2.line(netFrame, partA, partB, (0, 255, 0), 1)
                            cv2.circle(netFrame, partA, 1, (255, 0, 0), thickness=2, lineType=cv2.FILLED)
                    
                        
                #print("All Combos  = ", list(itertools.combinations(POSE_PAIRS, 2)))
                
                cv2.imshow('Output-Skeleton', frame)
                cv2.imshow('NET', netFrame) 
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

print("Hello World!!")
time.sleep(1)
print(cv2.useOptimized())
live_video()