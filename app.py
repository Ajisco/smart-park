import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the model outside of the function to avoid re-loading for each prediction
model = tf.keras.models.load_model('models/model_1.h5')

def preprocess_single_image(image_path, model):
    IMG_SIZE = 150
    dataset_labels = ['Free', 'Full']
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    img_pred = dataset_labels[np.argmax(prediction)]
    confidence = str(round(np.max(prediction) * 100, 2)) + '%'
    return img_pred, confidence

def display_home():
    st.header("Welcome to ParkAI!")
    st.write("ParkAI is a cutting-edge system that leverages deep learning to determine if a parking space is free or occupied. Check out some sample parking spots below!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/img1.jpg", caption="Parking Spot 1", use_column_width=True, clamp=True)
    with col2:
        st.image("images/img2.jpg", caption="Parking Spot 2", use_column_width=True,  clamp=True)
    with col3:
        st.image("images/img3.jpg", caption="Parking Spot 3", use_column_width=True, clamp=True)
    
    # Use session state to manage page navigation
    if st.button("Go to Prediction", key="home_predict_button"):
        st.session_state.page = "predict"

def display_predict():
    st.header("Predict Parking Availability")
    uploaded_file = st.file_uploader("Choose a parking spot image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        image = st.image(uploaded_file, caption='Uploaded Parking Spot Image.', use_column_width=True, clamp=True, channels='RGB')
        # Predict only when the Predict button is clicked
        if st.button("Predict Now"):
            st.write("Hold tight, we're getting insights for you!")
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            prediction, confidence = preprocess_single_image("temp_image.jpg", model)
            if prediction == "Free":
                st.success(f"The parking spot looks {prediction} with {confidence} confidence!")
            else:
                st.warning(f"The parking spot seems to be {prediction} with {confidence} confidence!")

def main():
    st.title('ParkAI - Smart Car Park System')
    
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Use session state to manage navigation
    if st.sidebar.button("Home", key="sidebar_home_button"):
        st.session_state.page = "home"
    if st.sidebar.button("Predict", key="sidebar_predict_button"):
        st.session_state.page = "predict"

    if st.session_state.page == "home":
        display_home()
    elif st.session_state.page == "predict":
        display_predict()
                
if __name__ == "__main__":
    main()
