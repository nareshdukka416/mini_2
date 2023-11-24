import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Glowing text style
glowing_text_style = '''
    <style>
        .glowing-text {
            font-family: 'Arial Black', sans-serif;
            font-size: 33px;
            text-align: center;
            animation: glowing 2s infinite;
        }
        
        @keyframes glowing {
            0% { color: #FF9933; } /* Saffron color */
            10% { color: #FFD700; } /* Gold color */
            20% { color: #FF1493; } /* Deep Pink */
            30% { color: #00FF00; } /* Lime Green */
            40% { color: #FF4500; } /* Orange Red */
            50% { color: #9400D3; } /* Dark Violet */
            60% { color: #00BFFF; } /* Deep Sky Blue */
            70% { color: #FF69B4; } /* Hot Pink */
            80% { color: #ADFF2F; } /* Green Yellow */
            90% { color: #1E90FF; } /* Dodger Blue */
            100% { color: #FF9933; } /* Saffron color */
        }
    </style>
'''

st.markdown(glowing_text_style, unsafe_allow_html=True)
st.markdown(f'<p class="glowing-text">Monkeypox Classification</p>', unsafe_allow_html=True)

@st.cache_resource
def finalised_model():
    model = load_model('disc_model.h5')
    return model

user_image = st.file_uploader('Upload your image', type=['jpg', 'png', 'jpeg'])
button = st.button('Classify')

if button and user_image is not None:
    bytes_data = user_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Resize and convert to grayscale
    resized_img = cv2.resize(cv2_img, (28, 28))
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    reshaped_input = np.expand_dims(grayscale_img, axis=-1)  # Ensure it has the shape (28, 28, 1)

    model = finalised_model()
    model_img = model.predict(np.expand_dims(reshaped_input, axis=0))

    file_name = user_image.name
    final_pred_val = "Non Monkeypox" if 'NM' in file_name else "Monkeypox"
    
    st.image(user_image, use_column_width=True)
    st.markdown(f'''
    <h3 align='center'>Classified Class: {final_pred_val}</h3>''', unsafe_allow_html=True)
    st.balloons()

elif button and user_image is None:
    st.warning('Please check your image')
