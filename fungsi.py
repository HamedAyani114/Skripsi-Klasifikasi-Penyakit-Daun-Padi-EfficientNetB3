from keras.utils import load_img, img_to_array
from tensorflow import keras
import streamlit as st
import numpy as np
import os


glob_input_kelas = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
path_model = os.path.join("models", "model_EfficientNetB3-FTAll_fold_3.h5")
model = keras.models.load_model(path_model)


def get_ImagePredict():
    img = load_img(st.session_state.data_image_predict, target_size=(300, 300))
    X_test = np.array([img_to_array(img)])
    result = model.predict(X_test)
    return result


def show_result():
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'> Prediksi Gambar </h1>",
        unsafe_allow_html=True,
    )

    radiopredict = st.radio(
        "Pilih Salah Satu",
        ("Upload Gambar", "Ambil Gambar dari Webcam"),
        key="radiopredict",
    )
    if radiopredict == "Upload Gambar":
        image_predict = st.file_uploader(
            "Upload Gambar",
            accept_multiple_files=False,
            key="data_image_predict",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )
    elif radiopredict == "Ambil Gambar dari Webcam":
        image_predict = st.camera_input(
            "Ambil Gambar dari Webcam",
            key="data_image_predict",
        )

    if image_predict:
        st.markdown("<h4>Image Predict</h4>", unsafe_allow_html=True)
        st.image(image_predict)
        st.markdown("<h4>Hasil</h4>", unsafe_allow_html=True)

        result = get_ImagePredict()
        y_pred = np.argmax(result, axis=1)
        y_pred_class = glob_input_kelas[y_pred[0]]

        st.success(
            "Gambar ini termasuk ke dalam kelas: %s - Probabilitas : %.3f"
            % (y_pred_class, result[0][y_pred] * 100)
        )
        for probability, kelas in zip(result[0], list(glob_input_kelas)):
            st.write("Kelas: %s - Probabilitas : %.3f" % (kelas, probability * 100))
            st.progress(int(probability * 100))
    else:
        st.info("Masukkan Gambar untuk prediksi", icon="ℹ️")
