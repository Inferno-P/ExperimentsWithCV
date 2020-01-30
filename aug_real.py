import cv2 
import numpy as np
from matplotlib import pyplot as plt


def live_video(camera_port=0):
        """
        Opens a window with live video.
        :param camera:
        :return:
        """

        video_capture = cv2.VideoCapture(camera_port)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            
            
            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            frame_v1 = frame 
            gray_frame_v1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('Video', frame)
            
            
            faces = face_cascade.detectMultiScale(gray_frame_v1, 1.3, 5)
            for (x,y,w,h) in faces:
                img_face = cv2.rectangle(frame_v1,(x,y),(x+w,y+h),(0,0,255),1)
                roi_gray = gray_frame_v1[y:y+h, x:x+w]
                roi_color = img_face[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Stay withing the face frame for the eyes
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),1)

            
            ### Initiate ORB detector
            orb = cv2.ORB_create()
            edges = cv2.Canny(frame,100,200) 

            # find the keypoints with ORB
            kp = orb.detect(frame, None)

            # compute the descriptors with ORB
            kp, des = orb.compute(frame, kp)
            
            # Finding the contours
            edges_2 = edges
            contours = cv2.findContours(edges_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print("Number of Contours found = " + str(len(contours)))
            
            
            # draw only keypoints location,not size and orientation
            img2 = cv2.drawKeypoints(frame, kp, frame, color=(0,255,0), flags=0)
            img2_edges = cv2.drawKeypoints(edges, kp, edges, color=(0,255,0), flags=0)
            
            
            cv2.imshow("KeyPoints", img2)
            cv2.imshow("KeyPoints with edges ", img2_edges)
            cv2.imshow("Face and Eye detection ",frame_v1)
            cv2.imshow("Contour Detection ",edges_2 )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


img = cv2.imread("E:\\Augmented\\scene.png",cv2.IMREAD_GRAYSCALE)
print("Hello World!!")
live_video()