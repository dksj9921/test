import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import numpy as np
from keras.utils import img_to_array
from PIL import Image
from keras.preprocessing import image
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import keras



st.title("Image Classification")
st.markdown('''


''')
file = st.file_uploader('이미지를 올려주세요.', type=['jpg','png'])

if file is None : 
    st.markdown('''### 이미지를 먼저 올려주세요.''')
else :
    bgd = Image.open(file)
    bgd_vector = np.asarray(img_to_array(bgd))
    bgd_vector = bgd_vector/255
    # print(bgd_vector)
 
    w, h = bgd.size
    s = min(w, h)
    y = (h - s) // 2
    x = (w - s) // 2
    
    print(w, h, x, y, s)
    bgd_vector_pad = bgd.crop((x, y, x+s, y+s))
    # 4-tuple defining the left, upper, right, and lower pixel coordinate

                               
    #이미지 형태 확인 

    model = VGG16(weights='imagenet', include_top=True)

    target_size = 224
    img = tf.image.resize(bgd_vector_pad, (target_size, target_size))
    
    img_batch = preprocess_input(img)

    img_batch = img_to_array(img)

    pre_processed = np.expand_dims(img_batch, axis=0)

   
    y_preds = model.predict(pre_processed)
    np.set_printoptions(suppress=True, precision=10)
    y_preds.shape
    
    y_preds = decode_predictions(y_preds, top=11)

    y_preds
    








