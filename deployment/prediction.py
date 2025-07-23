import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('./src/final_model.keras')

# Mapping label
class_dict = {'dry': 0, 'normal': 1, 'oily': 2}
label_lookup = {v: k for k, v in class_dict.items()}

img_height = 224
img_width = 224

# Fungsi prediksi
def predict_image(file):
    img = tf.keras.utils.load_img(file, target_size=(img_height, img_width))
    x = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred_proba = model.predict(x)
    pred_class = np.argmax(pred_proba[0])
    pred_label = label_lookup[pred_class]
    return img, pred_label, pred_proba[0]

def run():
    st.title('Skin Type Classification')
    st.subheader('Upload gambar wajah dan klik tombol Predict')

    uploaded_file = st.file_uploader("Upload file gambar (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        if st.button('Predict'):
            with st.spinner("Memprediksi..."):
                img, label, proba = predict_image(uploaded_file)

            st.markdown(f"### Prediksi: **{label.upper()}**")
            st.markdown("#### Probabilitas:")
            for i, cls in enumerate(label_lookup.values()):
                st.write(f"- {cls}: {proba[i]:.2f}")

if __name__ == '__main__':
    run()
