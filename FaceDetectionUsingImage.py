
# https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# Face detection using Haar cascade
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html?highlight=cvtcolor

import cv2  # Open source computer vision library
			# It can process images and videos to identify objects, faces, or even the handwriting of a human.
import random

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Classifier ultimately means detector. Classifier classify something as a face


# Choose an image to detect faces in
img = cv2.imread('faced.png')

# Convert to grayscale(Black & white)
# cvtColor --- convert color
grayscaled_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # Documentation for cv2.cvtColor : https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
														# BGR is blue-green-red

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) # MultiScale is for detecting faces of all sizes or compositions
																	  # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
# Since returns coordinates of rectangles --- it is face_coordinates

# print(face_coordinates) 
#eg. [[122 556 771 961]] --> 122,556 is of upper left and 771 , 961 is of lower right
# [[147 82 246 246]] --> Creates a square with upper left pt (147 , 82) lower right pt (147 + 246 , 82 + 246) 246 is width and height

# Draw rectangles around faces
#cv2.rectangle(img , (147 , 82) , (147+246 , 82+246) , (0 , 255 ,0) , 2) # Use coloured image
																# First two are the coordinates of rectangle
																# (0 , 255 , 0) --> (B , G , R)
																# 2 --> Thickness of the rectangle																
for (x , y , w , h) in face_coordinates:
#cv2.rectangle(img , (x , y) , (x+w , y+h) , (random.randrange(256) , random.randrange(256) , random.randrange(256)) , 10) # randrange() will produce random colours
	cv2.rectangle(img , (x , y) , (x+w , y+h) , (0 , 255 , 0) , 5)



# Image pops up
cv2.imshow('Cute Rhugweda',img)
cv2.waitKey() # Image will stay until any key is pressed


