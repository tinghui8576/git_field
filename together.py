import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 高斯滤波核大小
blur_ksize = 5
# Canny边缘检测高低阈值
canny_lth = 50
canny_hth = 150
# 霍夫变换参数
rho = 1
theta = np.pi / 180
threshold = 1
min_line_len = 1
max_line_gap = 50


# Read in the camera
#cap = cv2.VideoCapture(0)
# Read in the video
cap = cv2.VideoCapture('line_circle.mp4')

try:
    def process_an_image( img, weight):
        # 1. 灰度化、滤波和Canny
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        #edges = cv2.Canny(blur_gray, canny_lth, canny_hth)
        # 2. 标记四个坐标点用于ROI截取
        ed_rows, ed_cols = blur_gray.shape
        # points = np.array([[(0, 0), (0, rows), (cols, rows), (cols, 0)]])
        rows = int(ed_rows/4)
        cols = int(ed_cols/3.9)
        cols2 = int(3*ed_cols/3.9)
        #seperate the points into left and right
        point_left = np.array([[(0, rows), (0, ed_rows), (cols, ed_rows), (cols, rows)]])
        point_right = np.array([[(ed_cols, rows), (ed_cols, ed_rows), (cols2, ed_rows), (cols2, rows)]])
        #point3 = np.array([[(0, 120), (0, rows), (150, rows), (150, 120)]])
        #points = [point1, point2]
        # points = np.array([[(0, rows), (460, 325), (520, 325), (cols, rows)]])
        # [[[0 540], [460 325], [520 325], [960 540]]]
        #ROI in both left and right lanes
        roi_edges_left = roi_mask(blur_gray, point_left)
        roi_edges_right = roi_mask(blur_gray, point_right)
        # 3. 霍夫直线提取in both left and right
        drawing_left, lines_left, cen_left_x1, cen_left_x2, cen_left_y1, cen_left_y2 = hough_lines(roi_edges_left, rho, theta, threshold, min_line_len, max_line_gap)  
        drawing_right, lines_right, cen_right_x1, cen_right_x2, cen_right_y1, cen_right_y2 = hough_lines(roi_edges_right, rho, theta, threshold, min_line_len, max_line_gap)
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
        cen_x1, cen_y1, cen_x2, cen_y2 = find_center(lines) 
	# 新建一副空白画布
        drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(drawing, lines)     # 画出直线检测结果
        #draw lane's center line
        return drawing, lines, cen_x1, cen_x2, cen_y1, cen_y2

    def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
        # if(lines.all()):
        #     return
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
            count =  count + 1        
        center_x1 = center_x1 / count
        center_x2 = center_x2 / count
        center_y1 = center_y1 / count
        center_y2 = center_y2 / count

	#give the central line you calculate back
        return center_x1, center_y1, center_x2, center_y2
    
    #use video to send the message of where to go 
    def message_from_video(weight, cen_x1):
        #decide the direction by comparing the x direction between 
	#the center point from video(weight/2) and one of the central point from central line of the path
	#when the central point from line is smaller than center point means need to turn left, and so on  
        if(int(weight/2) < cen_x1):
            print("R")
        elif(int(weight/2) > cen_x1):
            print("L")
        else :
            print("S")

except (ValueError, ZeroDivisionError,TypeError):
        pass

while(cap.isOpened()):
	# Find fps
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	if int(major_ver)  < 3 :
		fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	else :
		fps = cap.get(cv2.CAP_PROP_FPS)
	
	# Find OpenCV version
	ret, image = cap.read()

	# make two copies of the image
	# With one copy we'll extract only the pixels that meet our selection,
	# then we'll paint those pixels red in the original image to see our selection
	# overlaid on the original.
	color_select= np.copy(image)
	line_image = np.copy(image)
	height, weight, channel =image.shape

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
	
	if cv2.waitKey(int(1.0/float(fps)*1000)) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
