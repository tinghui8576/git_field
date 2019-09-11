import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the camera
#cap = cv2.VideoCapture(0)
# Read in the video
cap = cv2.VideoCapture('line_corner.mp4')

while(cap.isOpened()):
	ret, image = cap.read()
	#image = mpimg.imread('field.png')
    # Grab the x and y sizes and make two copies of the image
    # With one copy we'll extract only the pixels that meet our selection,
    # then we'll paint those pixels red in the original image to see our selection
    # overlaid on the original.
	ysize = image.shape[0]
	xsize = image.shape[1]
	color_select= np.copy(image)
	line_image = np.copy(image)

    # Define our color criteria
	red_threshold = 90
	green_threshold = 50
	blue_threshold = 50
	rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    
	# Define a triangle region of interest (Note: if you run this code,
	# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
	# you'll find these are not sensible values!!
	# But you'll get a chance to play with them soon in a quiz ;)
	left_bottom = [0,ysize]
	right_bottom = [xsize,ysize]
	apex = [xsize/5, ysize/5]

	fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 2)
	fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 2)
	fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
    
   	# Mask pixels below the threshold
	color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
			(image[:,:,1] < rgb_threshold[1]) | \
			(image[:,:,2] < rgb_threshold[2]) 
			#((image[:,:,0] > 150) and (image[:,:,1] > 150) and (image[:,:,2] > 150))
	# Find the region inside the lines
	XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
	region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
			(YY > (XX*fit_right[0] + fit_right[1])) & \
			(YY < (XX*fit_bottom[0] + fit_bottom[1]))
	#print(region_thresholds)
    	# Mask color selection
	color_select[color_thresholds] = [0,0,0]
    	# Find where image is both colored right and in the region
	line_image[~color_thresholds & region_thresholds] = [255,0,0]

	# Display our two output images
	cv2.imshow('frame',color_select)
	cv2.imshow('frame',line_image)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
