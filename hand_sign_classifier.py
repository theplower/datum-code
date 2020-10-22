import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


model = keras.models.load_model("handClassifier.h5")
cap=cv.VideoCapture(0)


class_names = ["ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa","jeem","kaaf",
               "khaa","la","laam","meem","nun","ra","saad","seen","sheen","ta","taa","thaa","thal",
               "toot","waw","ya","yaa","zay",]
arabicnames = ["ع","ال","ا","ب","د","ظ","ض","ف","ق","غ","ح","ه","ج","ك","خ","لا","ل","م","ن","ر","ص","س","ش","ت","ط","ذ","ة","و","ئ","ي","ز"]



while True:
    
	_,frame = cap.read()
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame = cv.flip(frame,1)
	#converting it to the form that is known to the model
	test=np.array(frame,dtype=np.float32)
	test = test/255
	test = test.reshape(1,64,64,1)

	#getting the sign meaning 
	prediction = model.predict(test)
	if np.amax(prediction) > 0.4:
		print(arabicnames[np.argmax(prediction)])
		print(class_names[np.argmax(prediction)])

