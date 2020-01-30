#iimporting packages 

import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(1)

# set the reolution
cap.set(3,640)
cap.set(4,480)

while True:

	ret,frame = cap.read() #capture thqe image 
	
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convert BGR to HSV 

	lower_yellow = np.array([25,70,120])
	upper_yellow = np.array([30,255,255])


	mask1 = cv2.inRange(hsv_frame,lower_yellow,upper_yellow) #mask for yellow
	yellow = cv2.bitwise_and(frame, frame, mask=mask1)
	cnts1 = cv2.findContours(mask1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnts1 = imutils.grab_contours(cnts1)

	for c in cnts1: 
		area = cv2.contourArea(c)
		if area > 1000:
			cv2.drawContours(frame, [c],-1,(255,0,0),3)
		M = cv2.moments(c)
		cx = int(M["m10"]/1)
		cy = int(M["m01"]/1)

		#qcv2.putText(frame,"Yellow", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX,2.5,(255,255,255),3)
		cv2.imshow("frame",frame)
		cv2.imshow("mask",yellow)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release() 
