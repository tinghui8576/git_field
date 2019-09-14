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
cap = cv2.VideoCapture('line_corner.mp4')

try:
    def process_an_image(img):
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
        point1 = np.array([[(0, rows), (0, ed_rows), (cols, ed_rows), (cols, rows)]])
        point2 = np.array([[(ed_cols, rows), (ed_cols, ed_rows), (cols2, ed_rows), (cols2, rows)]])
        #point3 = np.array([[(0, 120), (0, rows), (150, rows), (150, 120)]])
        points = [point1, point2]
        # points = np.array([[(0, rows), (460, 325), (520, 325), (cols, rows)]])
        # [[[0 540], [460 325], [520 325], [960 540]]]
        roi_edges = roi_mask(blur_gray, points)
        # 3. 霍夫直线提取
        drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)  
        #     # 4. 车道拟合计算
        #draw_lanes(drawing, lines)
        # 5. 最终将结果合在原图上
        result = cv2.addWeighted(img, 0.9, drawing, 0.2, 0)
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
        # 新建一副空白画布
        drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(drawing, lines)     # 画出直线检测结果
        return drawing, lines

    def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
        # if(lines.all()):
        #     return
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
		

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


	result = process_an_image(color_select)
	# Display our two output images
	cv2.imshow('frame',result)
	# cv2.imshow('origin',image)
	
	if cv2.waitKey(int(1.0/float(fps)*1000)) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
