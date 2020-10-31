# Import the required libraries
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load the model
model = keras.models.load_model("handClassifier.h5")

# Use the PC camera
cap=cv.VideoCapture(0)

# Define class names
class_names = ["ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa","jeem","kaaf",
               "khaa","la","laam","meem","nun","ra","saad","seen","sheen","ta","taa","thaa","thal",
               "toot","waw","ya","yaa","zay",]
arabicnames = ["ع","ال","ا","ب","د","ظ","ض","ف","ق","غ","ح","ه","ج","ك","خ","لا","ل","م","ن","ر","ص","س","ش","ت","ط","ذ","ة","و","ئ","ي","ز"]

# Open a window to show a real-time detection
cv.namedWindow('Window')

# Define the font
font = cv.FONT_HERSHEY_SIMPLEX

while True:
	# Read a frame from the camera
	_,frame = cap.read()
	# Convert the frame to grayscale (because the classifier can only classify grayscale images)
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	# Resize the frame to the specific shape of the classifier
	frame = cv.resize(frame, (64, 64))
	# Flip the frame to show it properly
	frame = cv.flip(frame,1)
	# Convert it to the form that is known to the model
	test=np.array(frame,dtype=np.float32)
	test = test/255
	test = test.reshape(1,64,64,1)

	# Get the sign meaning 
	prediction = model.predict(test)
	confidencePercent = np.amax(prediction, 1).item()
	if confidencePercent > 0.4:
		print(arabicnames[np.argmax(prediction)])
		print(class_names[np.argmax(prediction)])
		letter = class_names[np.argmax(prediction)]
		message = 'Letter: %s   Prob: %.2f' % (letter, confidencePercent * 100) + '%'
	else:
		message = 'Not sure enough'
	# Write the predicted class on top of the current frame
	cv.putText(frame, message, (20, 25), font, 0.75, (255, 0, 0), 2, cv.LINE_AA)
	cv.imshow('Window', frame)
	# The program is stopped when the 'q' button is pressed
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
