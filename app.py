import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ---- Load your trained model ----
model = load_model('banana_ripeness_model.keras')  # Make sure this file is in the same folder

# ---- App title ----
st.title("🍌 Banana Ripeness Classification")
st.write("Upload a banana image and see if it’s Unripe, Ripe, or Overripe.")

# ---- Upload image ----
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    class_names = ['Unripe', 'Ripe', 'Overripe']

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("### Predictions:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[i]*100:.2f}%")

    st.write(f"✅ **Predicted Ripeness:** {class_names[np.argmax(prediction)]}")