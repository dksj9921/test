import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import img_to_array
from tensorflow.image import pad_to_bounding_box
from tensorflow.image import central_crop
from tensorflow.image import resize


st.title("Image Classification")
st.markdown('''


''')
file = st.file_uploader('이미지를 올려주세요.', type=['jpg','png'])

if file is None : 
    st.markdown('''### 이미지를 먼저 올려주세요.''')
else :
    bgd = file
    bgd_vector = np.asarray(img_to_array(bgd), dtype = None)
    bgd_vector = bgd_vector/255
 
    #이미지 형태 확인 
    bgd_vector.shape
 
    #이미지 확인 
    plt.imshow(bgd_vector)
    st.plt.show()


print(tf.version)
 
 
#weight, include_top 파라미터 설정 
model = VGG16(weights='imagenet', include_top=True)



