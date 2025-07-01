import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk menghindari error saat model tidak ditemukan
tf.get_logger().setLevel('ERROR')

st.title("Deteksi Chest X-ray 📸")
st.header("Identifikasi apa yang ada di gambar berikut!")

# Daftar kelas (sesuaikan dengan model Anda)
CLASSES = ['Normal', 'Aktif', 'Laten']  # Contoh kelas, sesuaikan!

# Model selection
model_paths = {
    "Model 1 (VGG-19)": "tb_vgg19_100_updt.h5",
    "Model 2 (Xception)": "tb_xception_100.h5"
}

# Sidebar untuk pemilihan model
choose_model = st.sidebar.selectbox(
    "Pilih model yang ingin digunakan",
    (model_paths.keys())
)

# Load model dengan caching
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Memuat model yang dipilih
model = load_model(model_paths[choose_model])

# Menampilkan info kelas
if st.checkbox("Tampilkan kelas yang tersedia"):
    st.write("Kelas yang dapat diidentifikasi:")
    st.write(CLASSES)

# Preprocessing khusus untuk setiap model
def preprocess_image(image, model_type):
    img = Image.open(image).convert('RGB')  # Konversi ke RGB
    
    # Resize sesuai kebutuhan model
    if "VGG-19" in model_type:
        img = img.resize((224, 224))
    elif "Xception" in model_type:
        img = img.resize((299, 299))
    else:  # Untuk model CNN default
        img = img.resize((150, 150))
    
    img_array = np.array(img)
    
    # Normalisasi khusus model
    if "VGG-19" in model_type:
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    elif "Xception" in model_type:
        img_array = tf.keras.applications.xception.preprocess_input(img_array)
    else:
        img_array = img_array / 255.0
    
    return np.expand_dims(img_array, axis=0)

# Upload gambar
uploaded_file = st.file_uploader(
    label="Upload gambar chest x-ray",
    type=["png", "jpeg", "jpg"]
)

if not uploaded_file:
    st.warning("Silahkan masukkan gambar.")
    st.stop()
else:
    st.image(uploaded_file, use_column_width=True)
    
    if st.button("Prediksi"):
        if model is None:
            st.error("Model belum dimuat dengan benar!")
            st.stop()
            
        with st.spinner("Sedang memproses..."):
            try:
                # Preprocessing
                processed_image = preprocess_image(uploaded_file, choose_model)
                
                # Prediksi
                prediction = model.predict(processed_image)
                
                # Ambil hasil prediksi
                predicted_class = CLASSES[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Tampilkan hasil
                st.success(f"Hasil Prediksi: **{predicted_class}**")
                st.metric(label="Tingkat Kepercayaan", value=f"{confidence:.2f}%")
                
                # Tampilkan grafik probabilitas
                st.bar_chart({
                    'Probabilitas': prediction[0]
                }, use_container_width=True)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")