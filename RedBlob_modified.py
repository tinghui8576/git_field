# Standard imports
import cv2
import numpy as np
from IPython import embed

# Read image

while True:
    frame = cv2.imread('BlobTest.jpg')

    canvas = frame.copy()

    lower = (0, 0, 0)
    upper = (200, 200, 200)
    mask = cv2.inRange(frame, lower, upper)
    try:
        # NB: using _ as the variable name for two of the outputs, as they're not used
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if c.shape[0] > 20]
        print(canvas)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

        for c in contours:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(canvas, center, 2, (0,0,255), -1)

    except (ValueError, ZeroDivisionError) as e:
        print(e)
        # pass

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('canvas', canvas)

    while cv2.waitKey(1) & 0xFF != ord('q'):
        pass
# cam.release()
cv2.destroyAllWindows()
