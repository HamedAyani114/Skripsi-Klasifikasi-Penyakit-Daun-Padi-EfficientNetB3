import streamlit as st
import fungsi as f


def main():
    st.markdown(
        "<h1 style='text-align:center'>KLASIFIKASI PENYAKIT DAUN PADI MENGGUNAKAN METODE DEEP LEARNING BERBASIS ARSITEKTUR EFFICIENNETB3 DENGAN TRANSFER LEARNING</h1>",
        unsafe_allow_html=True,
    )
    # navbar menu penjelasan dan menu prediksi
    menu = st.sidebar.radio("Menu", ("Penjelasan", "Prediksi"))
    if menu == "Penjelasan":
        st.markdown(
            """
            <h1 style='text-align:center;'>Penjelasan</h1>
            <p>Penelitian ini bertujuan untuk mengklasifikasikan penyakit daun padi menggunakan metode deep learning berbasis arsitektur EfficientNetB3 dengan transfer learning. 
            Penyakit daun padi yang diklasifikasikan adalah BrownSpot, Healthy, Hispa, dan LeafBlast. 
            Dataset yang digunakan merupakan data publik Kaggle oleh Shayan Riyaz total 3355 data citra dengan kelas BrownSpot, Healthy, Hispa dan LeafBlast yang masih belum seimbang, sehingga dilakukan oversampling augmentasi. Hasil penelitian model EfficientNetB3 dengan transfer learning fine-tuning dengan unfreeze semua layer berhasil mendapatkan performa evaluasi yang paling tinggi dengan akurasi pada data training 98,66%, testing 93,58% dan nilai AUC 0,9736, sedangkan pada model tanpa transfer learning mendapatkan akurasi training 59,40%, testing 77,47% dan AUC 0,8089.
            </p>
            """,
            unsafe_allow_html=True,
        )
        # image dataset
        st.image("images/dataset.png", use_column_width=True)
    else:
        f.show_result()

if __name__ == "__main__":
    main()
