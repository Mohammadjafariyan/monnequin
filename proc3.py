import cv2
import numpy as np
import random
import string

val = input("Enter file name: ") 

# Load the image
image = cv2.imread('D:/Temp/clothes/'+val)
output = image.copy()

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Create a mask for inpainting
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Draw rectangles around the faces on the mask
for (x, y, w, h) in faces:
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

# Apply inpainting to remove the faces
output = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

# Display the original and output images
cv2.imshow('Original Image', image)
cv2.imshow('Output Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()