import streamlit as st
import time
import datetime
import numpy as np
from PIL import Image
from preprocessing import preprocess_image, resize_image_for_preview
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import io

# Fungsi untuk menghindari error saat model tidak ditemukan
tf.get_logger().setLevel('ERROR')

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Chest X-ray",
    page_icon="📸",
    layout="wide"
)

# Inisialisasi session state untuk navigasi
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# Fungsi untuk mengganti halaman
def change_page(page_name):
    st.session_state.current_page = page_name

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
if st.sidebar.button("🔍 Deteksi", use_container_width=True):
    change_page("detection")

# Halaman Deteksi
if st.session_state.current_page == "detection":
    st.title("🔍 Deteksi Chest X-ray")
    st.markdown("---")
    
    # Daftar kelas
    CLASSES = ['Normal', 'Aktif', 'Laten']
    
    # Model selection
    model_paths = {
        "Model 1 (VGG-19)": "model_/tb_vgg19_100_updt.h5",
        # "Model 1 (VGG-19)": "model/vgg19/vgg19_scene3_100.h5"
        "Model 2 (Xception)": "model_/tb_xception_100.h5",
        # "Model 2 (Xception)": "model/xception/xception_scene2_200.h5",
        "Model 3 (VGG-16)": "model/vgg16/vgg16_scene1.pth",
        "Model 4 (CNN)": "model/cnn/cnn_scene3.keras"
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
        img = Image.open(image).convert('RGB')
        
        if "VGG-19" in model_type:
            img = img.resize((224, 224))
        elif "Xception" in model_type:
            img = img.resize((299, 299))
        else:
            img = img.resize((224, 224))
        
        img_array = np.array(img)
        
        if "VGG-19" in model_type:
            img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
        elif "Xception" in model_type:
            img_array = tf.keras.applications.xception.preprocess_input(img_array)
        else:
            img_array = img_array / 255.0
        
        return np.expand_dims(img_array, axis=0)
    
    # Fungsi untuk resize image untuk preview
    def resize_image_for_preview(image, model_type):
        img = Image.open(image).convert('RGB')
        
        if "VGG-19" in model_type:
            img = img.resize((224, 224))
        elif "Xception" in model_type:
            img = img.resize((299, 299))
        else:
            img = img.resize((224, 224))
        
        return img
    
    # Fungsi untuk prediksi batch
    def predict_batch(images, model, model_type):
        results = []
        
        for i, image in enumerate(images):
            try:
                # Preprocessing
                processed_image = preprocess_image(image, model_type)
                
                # Prediksi
                prediction = model.predict(processed_image, verbose=0)
                
                # Ambil hasil
                predicted_class = CLASSES[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                results.append({
                    'filename': image.name,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'normal_prob': prediction[0][0] * 100,
                    'aktif_prob': prediction[0][1] * 100,
                    'laten_prob': prediction[0][2] * 100
                })
                
            except Exception as e:
                results.append({
                    'filename': image.name,
                    'predicted_class': 'Error',
                    'confidence': 0,
                    'normal_prob': 0,
                    'aktif_prob': 0,
                    'laten_prob': 0,
                    'error': str(e)
                })
        
        return results

# Tab untuk single dan batch detection
tab1 = st.tabs(["🔍 Single Detection", "📸 Multi Detection"])
with tab1:
        st.subheader("Deteksi Single Image")
        # Upload gambar single
        uploaded_file = st.file_uploader(
            label="Upload gambar chest x-ray",
            type=["png", "jpeg", "jpg"],
            key="single_upload"
        )
        
        if not uploaded_file:
            st.warning("Silahkan masukkan gambar.")
        else:
            # Toggle untuk memilih tampilan
            view_mode = st.radio(
                "Pilih tampilan:",
                ["Original", "Resized", "Kedua"],
                horizontal=True
            )
            
            if view_mode == "Original":
                st.subheader("Gambar Input (Original):")
                st.image(uploaded_file, use_column_width=True)
                
            elif view_mode == "Resized":
                st.subheader(f"Gambar Resized ({choose_model}):")
                resized_img = resize_image_for_preview(uploaded_file, choose_model)
                original_img = Image.open(uploaded_file)
                st.image(resized_img, use_column_width=True)
                st.caption(f"Original: {original_img.size[0]}x{original_img.size[1]} → Resized: {resized_img.size[0]}x{resized_img.size[1]}")
            
            else:  # Kedua
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Gambar Input (Original):")
                    st.image(uploaded_file, use_column_width=True)
                    
                with col2:
                    st.subheader(f"Gambar Resized ({choose_model}):")
                    resized_img = resize_image_for_preview(uploaded_file, choose_model)
                    original_img = Image.open(uploaded_file)
                    st.image(resized_img, use_column_width=True)
                    st.caption(f"Original: {original_img.size[0]}x{original_img.size[1]} → Resized: {resized_img.size[0]}x{resized_img.size[1]}")
            
            # Button untuk memulai prediksi
            if st.button("🔍 Mulai Prediksi", type="primary", use_container_width=True):
                if model is None:
                    st.error("Model belum dimuat dengan benar!")
                else:
                    # Catat waktu mulai
                    start_time = time.time()
                    start_datetime = datetime.datetime.now()
                    
                    with st.spinner("Sedang memproses..."):
                        try:
                            # Preprocessing
                            preprocess_start = time.time()
                            processed_image = preprocess_image(uploaded_file, choose_model)
                            preprocess_time = time.time() - preprocess_start
                            
                            # Prediksi
                            predict_start = time.time()
                            prediction = model.predict(processed_image)
                            predict_time = time.time() - predict_start
                            
                            # Total waktu
                            total_time = time.time() - start_time
                            
                            # Ambil hasil prediksi
                            predicted_class = CLASSES[np.argmax(prediction)]
                            confidence = np.max(prediction) * 100
                            
                            # Tampilkan hasil
                            st.success(f"Hasil Prediksi: **{predicted_class}**")
                            st.metric(label="Tingkat Kepercayaan", value=f"{confidence:.2f}%")
                            
                            # Tampilkan informasi kecepatan
                            st.subheader("⚡ Informasi Kecepatan:")
                            
                            col_speed1, col_speed2, col_speed3 = st.columns(3)
                            
                            with col_speed1:
                                st.metric(
                                    label="Total Waktu",
                                    value=f"{total_time:.3f}s",
                                    delta=f"{(total_time*1000):.1f}ms"
                                )
                            
                            with col_speed2:
                                st.metric(
                                    label="Preprocessing",
                                    value=f"{preprocess_time:.3f}s",
                                    delta=f"{(preprocess_time*1000):.1f}ms"
                                )
                            
                            with col_speed3:
                                st.metric(
                                    label="Prediksi",
                                    value=f"{predict_time:.3f}s",
                                    delta=f"{(predict_time*1000):.1f}ms"
                                )
                            
                            # Informasi detail waktu
                            st.info(f"""
                            📅 **Waktu Eksekusi**: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}
                            ⏱️ **Durasi Total**: {total_time:.3f} detik ({total_time*1000:.1f} ms)
                            �� **Preprocessing**: {preprocess_time:.3f} detik ({preprocess_time*1000:.1f} ms)
                            🤖 **Prediksi Model**: {predict_time:.3f} detik ({predict_time*1000:.1f} ms)
                            """)
                            
                            # Tampilkan grafik probabilitas
                            st.subheader("Probabilitas per Kelas:")
                            prob_data = {
                                'Kelas': CLASSES,
                                'Probabilitas': prediction[0]
                            }
                            st.bar_chart(prob_data, x='Kelas', y='Probabilitas', use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {e}")