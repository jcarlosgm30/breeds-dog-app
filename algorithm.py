import numpy as np
import cv2                
from glob import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from extract_bottleneck_features import extract_Resnet50
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D

class modelBreedsDogs(object):
    def __init__(self, weights="imagenet"):
        # bottleneck_features = np.load('/data/bottleneck_features/DogResnet50Data.npz')
        # self.train_ResNet50 = bottleneck_features['train']
        self.ResNet50_model = self.model(weights)
        self.dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    
    def model(self, weights):
        ResNet50_model= ResNet50(weights=weights)
        ResNet50_model = Sequential()
        ResNet50_model.add(GlobalAveragePooling2D(input_shape=[1, 2048]))
        ResNet50_model.add(Dense(128, activation='sigmoid'))
        ResNet50_model.add(Dense(133, activation='softmax'))
        ResNet50_model.summary()
    
        return ResNet50_model

    def path_to_tensor(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)
    
    def predict_Labels(self, img_path):
        img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def dog_detector(self, img_path):
        prediction = self.predict_Labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    def predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(self.path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.ResNet50_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]  

    def detecting_image(self, img_path):
        if self.face_detector(img_path):
            return "It is a human face! he/she looks like a {}".format(self.predict_breed(img_path))
        if self.dog_detector(img_path):
            return "The predicted dog breed is {}".format(self.predict_breed(img_path))
        else:
            return "There are not a dog or human face in the image"

model = modelBreedsDogs()

image_path='images/dog1.jpg'
model.detecting_image(image_path)
