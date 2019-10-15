import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import serial
import time
import struct

# 指定通訊埠名稱
COM_PORT = '/dev/ttyACM0'
# 設定傳輸速率
BAUD_RATES = 9600
# 初始化序列通訊埠
ser = serial.Serial(COM_PORT, BAUD_RATES)  
# communication treshold 
tre = 10

# 高斯滤波核大小
blur_ksize = 15
# Canny边缘检测高低阈值
canny_lth = 50
canny_hth = 150
# 霍夫变换参数
rho = 1
theta = np.pi / 180
threshold = 1
min_line_len = 20
max_line_gap = 15
cc = 0
sm = 5
smt = [0,0,0,0,0,0,0,0,0,0]

cc = 0
sm = 5
smt = [0,0,0,0,0,0,0,0,0,0]


# Read in the camera
<<<<<<< HEAD
cap = cv2.VideoCapture(1)
=======
cap = cv2.VideoCapture(0)
>>>>>>> 9948e256bc058f446e97c649f2268ffd6ba72ad9
# Read in the video
#cap = cv2.VideoCapture('right1.avi')

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
try:
	def process_an_image( img, weight):
		# 1. 灰度化、滤波和Canny
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		#edges = cv2.Canny(blur_gray, canny_lth, canny_hth)
		# 2. 标记四个坐标点用于ROI截取
		ed_rows, ed_cols = gray.shape
		# points = np.array([[(0, 0), (0, rows), (cols, rows), (cols, 0)]])
		# rows = int(ed_rows/4)
		# cols = int(ed_cols/3.9)
		# cols2 = int(3*ed_cols/3.9)
        	#seperate the points into left and right
		# point_left = np.array([[(0, rows), (0, ed_rows), (cols, ed_rows), (cols, rows)]])
		# point_right = np.array([[(ed_cols, rows), (ed_cols, ed_rows), (cols2, ed_rows), (cols2, rows)]])
		#point3 = np.array([[(0, 120), (0, rows), (150, rows), (150, 120)]])
		#points = [point1, point2]
		# points = np.array([[(0, rows), (460, 325), (520, 325), (cols, rows)]])
		# [[[0 540], [460 325], [520 325], [960 540]]]
		#ROI in both left and right lanes
		# roi_edges_left = roi_mask(blur_gray, point_left)
		# roi_edges_right = roi_mask(blur_gray, point_right)
		# 3. 霍夫直线提取in both left and right
		drawing_left, lines_left, cen_left_x1, cen_left_x2, cen_left_y1, cen_left_y2 = hough_lines(gray, rho, theta, threshold, min_line_len, max_line_gap)  
		drawing_right, lines_right, cen_right_x1, cen_right_x2, cen_right_y1, cen_right_y2 = hough_lines(gray, rho, theta, threshold, min_line_len, max_line_gap)
		cen_x1 = (cen_left_x1 + cen_right_x1)/2
		cen_x2 = (cen_left_x2 + cen_right_x2)/2
		cen_y1 = (cen_left_y1 + cen_right_y1)/2
		cen_y2 = (cen_left_y2 + cen_right_y2)/2
	#use the center point of the video and the central line of the path to control motor by sending message
		message_from_video(weight, int(cen_x1))
		# 5. 最终将结果合在原图上
		#also let both right and left lanes put into one pic
		drawing = cv2.addWeighted(drawing_left, 1, drawing_right, 1, 0)
		cv2.line(drawing, (int(cen_x1), int(cen_y1)), (int(cen_x2), int(cen_y2)), (255,0,0),5)
		result = cv2.addWeighted(img, 0.9, drawing, 0.7, 0)
		return result
		#return roi_edges

	def roi_mask(img, corner_points):
		# 创建掩膜
		mask = np.zeros_like(img)
		cv2.fillPoly(mask, corner_points, 255)
		masked_img = cv2.bitwise_and(img, mask)
		return masked_img

	def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
		# 统计概率霍夫直线变换
		lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
		if(lines is None):
			return 0,0,0,0,0,0
		cen_x1, cen_y1, cen_x2, cen_y2 = find_center(lines) 
	# 新建一副空白画布
		drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
		draw_lines(drawing, lines)     # 画出直线检测结果
		#draw lane's center line
		return drawing, lines, cen_x1, cen_x2, cen_y1, cen_y2

	def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	# a function to find the center of the line
	def find_center(lines):
		#define the center points and the number of hough lines you have
		center_x1 = 0
		center_x2 = 0
		center_y1 = 0
		center_y2 = 0
		count = 0

		#add all the position in the hough lines and find out the center 
		for line in lines:
			for x1, y1 ,x2, y2 in line:
				center_x1 = x1 + center_x1
				center_x2 = x2 + center_x2
				center_y1 = y1 + center_y1
				center_y2 = y2 + center_y2
			count += 1        
		center_x1 /= count
		center_x2 /=  count
		center_y1 /=  count
		center_y2 /=  count

		#give the central line you calculate back
		return center_x1, center_y1, center_x2, center_y2
	
	#use video to send the message of where to go 
	def message_from_video(weight, cen_x1):
		#decide the direction by comparing the x direction between 
		#the center point from video(weight/2) and one of the central point from central line of the path
		#when the central point from line is smaller than center point means need to turn left, and so on  
		# ser.write(3*((int(weight/2) - cen_x1)))
<<<<<<< HEAD
		#print(((int(weight/2) - cen_x1)))
=======
		
>>>>>>> 9948e256bc058f446e97c649f2268ffd6ba72ad9
		move = (int(weight/2) - cen_x1)

		if (move > 127):
			move = 127
		elif (move < -127):
			move = -127
<<<<<<< HEAD
		smooth_deliever(move)

		#if ((move > tre) or (move < -tre)):
		#	ser.write(str(move).encode('ascii'))
		#	time.sleep(1)
			# print(str(move).encode('ascii'))
		# while ser.in_waiting:
		# 		mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
		# 		print('控制板回應：', mcu_feedback)
		# 		break

=======
		smooth_deliver(move)
		
>>>>>>> 9948e256bc058f446e97c649f2268ffd6ba72ad9
		# if(int(weight/2) < cen_x1):
		# 	print("R")
		# elif(int(weight/2) > cen_x1):
		# 	print("L")
		# else :
		# 	print("S")
        def smooth_deliever(move):
            global cc, sm, smt
            if(cc<sm):
                smt[cc,=move]
                cc+=1
            else:
                k = 0
                cc = 0
                for i in range(sm):
                    k+=smt[i]
                k /= sm
                k = int(k)
                print(k)
                if ((k > tre) or (k < -tre)):
                    ser.write(str(k).encode('ascii'))

	def smooth_deliver(move):
		global cc, sm, smt
		if(cc < sm):
			smt[cc] = move
			cc+=1
		else:
			k = 0
			cc = 0
			for i in range(sm):
				k += smt[i]
			k /= 10
			k = int(k)
			print(k)
			if ((k > tre) or (k < -tre)):
				ser.write(str(k).encode('ascii'))
			# time.sleep(1)
			# while ser.in_waiting:
			# 		mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
			# 		print('控制板回應：', mcu_feedback)
			# 		break

except (ValueError, ZeroDivisionError,TypeError):
		pass

#ret, image = cap.read()
while(cap.isOpened()):	
	# Find fps
	fps = cap.get(cv2.CAP_PROP_FPS)
	# print('fps', fps)	
	# Find OpenCV version
	ret, image = cap.read()
	image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 1)

	# make two copies of the image
	# With one copy we'll extract only the pixels that meet our selection,
	# then we'll paint those pixels red in the original image to see our selection
	# overlaid on the original.
	color_select= np.copy(image)
	line_image = np.copy(image)
	height, weight, channel = image.shape
	
	# Define our color criteria
	HMi_threshold = 10
	HMa_threshold = 156
	S_threshold = 43
	V_threshold = 46
	HSV_threshold = [HMi_threshold, HMa_threshold, S_threshold, V_threshold]
	# Mask pixels below the threshold
	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	H, S, V = cv2.split(HSV)
	color_thresholds = ((H > HSV_threshold[0]) & (H < HSV_threshold[1])) | (S < HSV_threshold[2]) | (V < HSV_threshold[3])
	# Mask color selection
	color_select[color_thresholds] = [0,0,0]
	color_select[~color_thresholds] = [255,255,255]
	# Find where image is both colored right and in the region
	line_image[~color_thresholds] = [255,0,0]

	result = process_an_image(color_select, weight)
	# Display our two output images
	cv2.imshow('frame',result)
	#cv2.imshow('origin',image)

	if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
