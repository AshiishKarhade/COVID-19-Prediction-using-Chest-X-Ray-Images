import numpy as np
import pandas as pd
import keras
from keras.models import load_model
import streamlit as st
from PIL import Image
from keras import backend as K

st.title('COVID-19 Predictor')

small_model = 'model.h5'
big_model = 'resnet152v2e5v1.h5'
model = load_model(big_model)

disease = {0:'COVID', 1:'Lung_Opacity', 2:'Normal', 3:'Viral Pneumonia'}

def predict(image_loc):
    sub = Image.open(image_loc)
    sub = sub.resize((224, 224))
    sub = np.array(sub)
    #print(sub.shape)
    #sub = sub.reshape(-1, 224, 224, 3)
    sub = np.expand_dims(sub, axis=0)
    preds = model.predict(sub)
    #print(preds)
    preds = preds.flatten()
    #print(preds)
    result = np.argmax(preds)
    #print(result)  
    return disease[result]


uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    st.write("")
    st.text("classifying...")
    with st.spinner():
        label = predict(uploaded_file)
    #print(label)
    st.header('The result is:-')
    st.subheader('%s' % (label))
