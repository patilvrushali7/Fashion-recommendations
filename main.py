import os
import pickle
import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors


st.title('Fashion Recommendation')

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
file_names = pickle.load(open('filenames.pkl', 'rb'))



def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(feature,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, metric='euclidean',algorithm='brute')
    neighbors.fit(feature_list)

    distance, indices = neighbors.kneighbors([feature])
    return indices


uploaded_file = st.file_uploader('Select a File')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
       display_img = Image.open(uploaded_file)
       st.image(display_img)
       feature = feature_extraction(os.path.join('uploads',uploaded_file.name),model)

       indices = recommend(feature,feature_list)
       col1,col2,col3,col4,col5 = st.columns(5)
       with col1:
           st.image(file_names[indices[0][0]])
       with col2:
           st.image(file_names[indices[0][1]])
       with col3:
           st.image(file_names[indices[0][2]])
       with col4:
           st.image(file_names[indices[0][3]])
       with col5:
           st.image(file_names[indices[0][4]])
    else:
        st.header('error')