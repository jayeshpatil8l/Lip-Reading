import os
import streamlit as st
import tensorflow as tf
# import imageio
from tensorflow.keras.backend import ctc_decode # type: ignore
from tensorflow.keras.models import Model # type: ignore
from utils import load_data, num_to_char
from model import build_model

input_shape = (75, 46, 140, 1)
vocab_size = num_to_char.vocabulary_size()

model = build_model(input_shape, vocab_size)

# model = tf.keras.models.load_model(filepath=os.path.join('.','models','model_checkpoint_after_50.keras'))

st.set_page_config(layout = 'wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipNet')
    st.info('Lip Reading Model using Deep Learning Techniques inspired by the LipNet Research Paper')

st.title("Lip Reading Application")
videos = os.listdir(os.path.join('..','Data','s1'))
selected_video = st.selectbox('Choose Video', videos, placeholder = "Select a Video")

col1, col2 = st.columns(2)

with col1:
    st.info('Below is the Selected Video in MP4 Format!')
    file_path = os.path.join('..','Data','s1',selected_video)
    os.system(f"ffmpeg -i {file_path} -vcodec libx264 temp_video.mp4 -y")

    converted_video = open('./temp_video.mp4', 'rb')
    video_bytes = converted_video.read()
    st.video(video_bytes)

with col2:
    # st.info('This is all the machine learning model sees while prediction')
    # video, labels = load_data(tf.convert_to_tensor(file_path))
    # imageio.mimsave('animation.gif', video, fps = 10)
    # st.image('./animation.gif', width = 400)

    st.info('This is the output of the machine learning model as tokens!')
    video, labels = load_data(tf.convert_to_tensor(file_path))
    yhat = model.predict(tf.expand_dims(video, axis = 0))
    decoded = ctc_decode(yhat, [75], greedy = True)[0][0].numpy()
    st.text(decoded)

    st.info("Decoded tokens to words!")
    converted_prediction = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
    st.text(converted_prediction)



