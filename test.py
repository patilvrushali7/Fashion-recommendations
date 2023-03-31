import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
file_names = pickle.load(open('filenames.pkl', 'rb'))

img = image.load_img('test/1549.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocess_img = preprocess_input(expanded_img_array)
result = model.predict(preprocess_img).flatten()
normalized_result = result / norm(result)
normalized_result

neighbors = NearestNeighbors(n_neighbors=6, metric='cosine')
neighbors.fit(feature_list)

distance, indices = neighbors.kneighbors([normalized_result])

for file in indices[0][1:6]:
    temp_img = cv2.imread((file_names[file]))
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitkey(0)
