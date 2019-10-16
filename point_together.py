import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import serial

# 指定通訊埠名稱
COM_PORT = '/dev/ttyACM0'
# 設定傳輸速率
BAUD_RATES = 9600
# 初始化序列通訊埠
ser = serial.Serial(COM_PORT, BAUD_RATES)
# communication treshold
tre = 10


# Read in the camera
#cap = cv2.VideoCapture(0)
# Read in the video
cap = cv2.VideoCapture('right1.avi')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
try:
	def message_from_video(cen_x, cen_y):
        #decide the direction by comparing the x direction between
        #the center point from video(weight/2) and one of the central point from central line of the path
        #when the central point from line is smaller than center point means need to turn left, and so on
        # ser.write(3*((int(weight/2) - cen_x1)))
        
        move = (int(weight/2) - cen_x1)

        if (move > 127):
            move = 127
        elif (move < -127):
            move = -127
        if ((move > tre) or (move < -tre)):
            ser.write(str(move).encode('ascii'))
        
except (ValueError, ZeroDivisionError,TypeError):
        pass
        
while(cap.isOpened()):	
	
 
 
	cv2.imshow('frame',result)
	

	if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
