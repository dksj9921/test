import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import img_to_array
from PIL import Image
# from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
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
 
    #이미지의 변경할 크기 설정 
    target_height = 4500
    target_width = 4500
    
    #현재 이미지의 크기 지정 
    source_height = bgd_vector.shape[0]
    source_width = bgd_vector.shape[1]
    
    #padding 실시 : pad_to_bounding_box 사용 
    bgd_vector_pad = tf.image.pad_to_bounding_box(bgd_vector, 
                                        int((target_height-source_height)/2), 
                                        int((target_width-source_width)/2), 
                                        target_height, 
                                        target_width)
                                        
    #이미지 형태 확인 
    bgd_vector_pad.shape

    model = VGG16(weights='imagenet', include_top=True)

    target_size = 224
    img = tf.image.resize(bgd_vector_pad, (target_size, target_size))
    
    np_img = img_to_array(img)
 
 
    # #4차원으로 변경 
    img_batch = np.expand_dims(np_img, axis=0)

    img_batch = preprocess_input(img_batch)
    




 
 
#weight, include_top 파라미터 설정 




