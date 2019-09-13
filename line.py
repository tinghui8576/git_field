import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the camera
#cap = cv2.VideoCapture(0)
# Read in the video
cap = cv2.VideoCapture('line_corner.mp4')

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

	# Display our two output images
	cv2.imshow('frame',color_select)
	#cv2.imshow('frame',line_image)
	
	if cv2.waitKey(int(1.0/float(fps)*1000)) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
