import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

from PIL import Image

size = (100, 100)
artists = ['Edgar Degas', 'Pablo Picasso', 'Pierre-Auguste Renoir', 'Vincent van Gogh']

def preprocess_img(img):
	'''
	This method processes the image into the correct expected shape in the model (28, 28). 
	'''
	img = np.array(img).astype('float32')/255
	img = cv2.resize(img, (100, 100))
	img = np.expand_dims(img, axis=0)
	'''img = image.resize(img, (100, 100))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	image = image / 255
	image = image.resize(size)
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)'''
	return img

def image_loader(image):
	''' 
	This method loads the image into a PyTorch tensor. 
	'''
	image = TF.to_tensor(image)
	image = image.unsqueeze(0)
	return image

class Predictor: 
	def __init__(self):
                self.model = load_model('result.h5')
    
	def predict(self, request):
		'''
		This method reads the file uploaded from the Flask application POST request, 
		and performs a prediction using the MNIST model. 
		'''
		f = request.files['image']
		image = Image.open(f)
		image = preprocess_img(image)
		#image = image_loader(image)
		prediction = self.model.predict(image)
		prediction_probability = np.amax(prediction)
		prediction_idx = np.argmax(prediction)
		return artists[prediction_idx.item()]
