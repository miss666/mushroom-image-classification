from predict import dist,Predict
import streamlit as st
from PIL import Image

import time
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Mushroom Classification App")
st.write("")
st.write("")
option = st.selectbox(
     'Choose the model you want to use?',
     ('EfficientNet-B0', 'MobileNet-V2', 'SE-ResNext101_32x4d'))
option1 = st.selectbox(
     'Select the image upload method',
     ('Upload an image', 'Take a picture'))
if option1=='Upload an image':
    file_up = st.file_uploader("Upload an image", type="jpg")
else:
    file_up = st.camera_input("Take a picture")

if file_up is None:
    st.write("")
    st.write("please chose a picture...")

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels,s = Predict(file_up,option)

    # print out the top 5 prediction labels with scores
    st.success('successful prediction')
    for i in labels:
        st.write("Prediction (index, name):", i[0], ",   Score: ", i[1])

    # print(t2-t1)
    # st.write(float(t2-t1))
    st.write("")
    st.metric("","Seconds:   "+str(s))
