################

# ADA PREVIEW IMAGE SEKALIGUS PREVIEW DUA IMAGE HASIL NYA OKE

##############
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
import datetime

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
if st.sidebar.button("🏠 Beranda", use_container_width=True):
    change_page("home")
if st.sidebar.button("🔍 Deteksi", use_container_width=True):
    change_page("detection")
if st.sidebar.button("ℹ️ Tentang", use_container_width=True):
    change_page("about")

# Halaman Beranda
if st.session_state.current_page == "home":
    st.title("🏠 Beranda - Deteksi Chest X-ray")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Selamat Datang!")
        st.write("""
        Aplikasi ini dirancang untuk membantu mendeteksi kondisi chest x-ray menggunakan 
        teknologi machine learning. Kami menggunakan model deep learning yang telah 
        dilatih untuk mengidentifikasi berbagai kondisi pada gambar chest x-ray.
        """)
        
        st.subheader("Fitur Utama:")
        st.write("""
        ✅ Deteksi kondisi Normal, Aktif, dan Laten  
        ✅ Dukungan multiple model (VGG-19 & Xception)  
        ✅ Interface yang user-friendly  
        ✅ Hasil prediksi yang akurat  
        """)
    
    with col2:
        st.image("image/icon_xray.png", width=400)
    
    st.markdown("---")
    st.info("💡 **Tips**: Gunakan tombol 'Deteksi' di sidebar untuk mulai menganalisis gambar chest x-ray Anda.")

# Halaman Deteksi
elif st.session_state.current_page == "detection":
    st.title("🔍 Deteksi Chest X-ray")
    st.markdown("---")
    
    # Daftar kelas
    CLASSES = ['Normal', 'Aktif', 'Laten']
    
    # Model selection
    model_paths = {
        # "Model 1 (VGG-19)": "model/tb_vgg19_100_updt.h5",
        "Model 1 (VGG-19)": "model/vgg19/vgg19_scene3_100.h5",
        # "Model 2 (Xception)": "model/tb_xception_100.h5",
        "Model 2 (Xception)": "model/xception/xception_scene2_200.h5",
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
    tab1, tab2 = st.tabs(["🔍 Single Detection", "📸 Multi Detection"])
    
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
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar Input (Original):")
                st.image(uploaded_file, use_column_width=True)
                
                # Tampilkan gambar yang sudah diresize
                st.subheader(f"Gambar Resized ({choose_model}):")
                resized_img = resize_image_for_preview(uploaded_file, choose_model)
                st.image(resized_img, use_column_width=True)
                
                # Tampilkan informasi ukuran
                original_img = Image.open(uploaded_file)
                st.caption(f"Original: {original_img.size[0]}x{original_img.size[1]} → Resized: {resized_img.size[0]}x{resized_img.size[1]}")
            
            with col2:
                st.subheader("Kontrol:")
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
                                🔧 **Preprocessing**: {preprocess_time:.3f} detik ({preprocess_time*1000:.1f} ms)
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
    
    with tab2:
        st.subheader("Deteksi Multiple Images")
        # Upload multiple gambar
        uploaded_files = st.file_uploader(
            label="Upload multiple gambar chest x-ray",
            type=["png", "jpeg", "jpg"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if not uploaded_files:
            st.warning("Silahkan upload beberapa gambar chest x-ray.")
            st.info("💡 **Tips**: Anda dapat memilih lebih dari satu file dengan menahan Ctrl (Windows) atau Cmd (Mac) saat memilih file.")
        else:
            st.success(f"✅ Berhasil upload {len(uploaded_files)} gambar")
                    
            # Tampilkan preview gambar dengan toggle untuk original/resized
            st.subheader("📷 Preview Gambar:")
            
            # Toggle untuk memilih tampilan
            view_mode = st.radio(
                "Pilih tampilan:",
                ["Original", "Resized", "Kedua"],
                horizontal=True
            )
            
            if view_mode == "Original":
                cols = st.columns(min(4, len(uploaded_files)))
                for idx, uploaded_file in enumerate(uploaded_files):
                    col_idx = idx % 4
                    with cols[col_idx]:
                        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
            
            elif view_mode == "Resized":
                cols = st.columns(min(4, len(uploaded_files)))
                for idx, uploaded_file in enumerate(uploaded_files):
                    col_idx = idx % 4
                    with cols[col_idx]:
                        resized_img = resize_image_for_preview(uploaded_file, choose_model)
                        original_img = Image.open(uploaded_file)
                        caption = f"{uploaded_file.name}\n{original_img.size[0]}x{original_img.size[1]} → {resized_img.size[0]}x{resized_img.size[1]}"
                        st.image(resized_img, caption=caption, use_column_width=True)
            
            else:  # Kedua
                for idx, uploaded_file in enumerate(uploaded_files):
                    with st.expander(f"📷 {uploaded_file.name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original:")
                            st.image(uploaded_file, use_column_width=True)
                        
                        with col2:
                            st.subheader(f"Resized ({choose_model}):")
                            resized_img = resize_image_for_preview(uploaded_file, choose_model)
                            original_img = Image.open(uploaded_file)
                            st.image(resized_img, use_column_width=True)
                            st.caption(f"{original_img.size[0]}x{original_img.size[1]} → {resized_img.size[0]}x{resized_img.size[1]}")

            # Button untuk memulai prediksi batch
            if st.button("🚀 Mulai Batch Prediction", type="primary", use_container_width=True):
                if model is None:
                    st.error("Model belum dimuat dengan benar!")
                else:
                    # Catat waktu mulai
                    batch_start_time = time.time()
                    batch_start_datetime = datetime.datetime.now()
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Prediksi batch
                    with st.spinner("Sedang memproses batch prediction..."):
                        results = predict_batch(uploaded_files, model, choose_model)
                    
                    # Total waktu batch
                    batch_total_time = time.time() - batch_start_time
                    
                    # Tampilkan hasil dalam tabel
                    st.subheader("📊 Hasil Batch Prediction:")
                    
                    # Buat DataFrame
                    df = pd.DataFrame(results)
                    
                    # Tampilkan tabel dengan styling
                    st.dataframe(
                        df,
                        column_config={
                            "filename": "Nama File",
                            "predicted_class": "Prediksi",
                            "confidence": st.column_config.NumberColumn(
                                "Tingkat Kepercayaan (%)",
                                format="%.2f%%"
                            ),
                            "normal_prob": st.column_config.NumberColumn(
                                "Normal (%)",
                                format="%.2f%%"
                            ),
                            "aktif_prob": st.column_config.NumberColumn(
                                "Aktif (%)",
                                format="%.2f%%"
                            ),
                            "laten_prob": st.column_config.NumberColumn(
                                "Laten (%)",
                                format="%.2f%%"
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Informasi kecepatan batch
                    st.subheader("⚡ Informasi Kecepatan Batch:")
                    
                    col_batch1, col_batch2, col_batch3, col_batch4 = st.columns(4)
                    
                    with col_batch1:
                        st.metric(
                            label="Total Waktu",
                            value=f"{batch_total_time:.3f}s",
                            delta=f"{(batch_total_time*1000):.1f}ms"
                        )
                    
                    with col_batch2:
                        avg_time_per_image = batch_total_time / len(uploaded_files)
                        st.metric(
                            label="Rata-rata per Gambar",
                            value=f"{avg_time_per_image:.3f}s",
                            delta=f"{(avg_time_per_image*1000):.1f}ms"
                        )
                    
                    with col_batch3:
                        images_per_second = len(uploaded_files) / batch_total_time
                        st.metric(
                            label="Kecepatan",
                            value=f"{images_per_second:.1f}",
                            delta="gambar/detik"
                        )
                    
                    with col_batch4:
                        st.metric(
                            label="Total Gambar",
                            value=len(uploaded_files),
                            delta="diproses"
                        )
                    
                    # Informasi detail batch
                    st.info(f"""
                    📅 **Waktu Eksekusi**: {batch_start_datetime.strftime('%d/%m/%Y %H:%M:%S')}
                    ⏱️ **Durasi Total**: {batch_total_time:.3f} detik ({batch_total_time*1000:.1f} ms)
                    📊 **Total Gambar**: {len(uploaded_files)} gambar
                    🚀 **Kecepatan**: {images_per_second:.1f} gambar/detik
                    ⏳ **Rata-rata per Gambar**: {avg_time_per_image:.3f} detik ({avg_time_per_image*1000:.1f} ms)
                    """)
                    
                    # Statistik ringkasan
                    st.subheader("📈 Statistik Ringkasan:")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_images = len(results)
                        st.metric("Total Gambar", total_images)
                    
                    with col2:
                        successful_predictions = len([r for r in results if r['predicted_class'] != 'Error'])
                        st.metric("Prediksi Berhasil", successful_predictions)
                    
                    with col3:
                        avg_confidence = np.mean([r['confidence'] for r in results if r['predicted_class'] != 'Error'])
                        st.metric("Rata-rata Kepercayaan", f"{avg_confidence:.2f}%")
                    
                    with col4:
                        most_common = df['predicted_class'].mode().iloc[0] if not df.empty else "N/A"
                        st.metric("Prediksi Terbanyak", most_common)
                    
                    # Grafik distribusi prediksi
                    st.subheader("📊 Distribusi Prediksi:")
                    
                    if not df.empty:
                        # Filter hasil yang berhasil
                        successful_df = df[df['predicted_class'] != 'Error']
                        
                        if not successful_df.empty:
                            # Bar chart untuk distribusi kelas
                            class_counts = successful_df['predicted_class'].value_counts()
                            st.bar_chart(class_counts)
                            
                            # Pie chart untuk distribusi
                            fig, ax = plt.subplots()
                            ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
                            ax.set_title('Distribusi Prediksi')
                            st.pyplot(fig)
                    
                    # Download hasil
                    st.subheader("💾 Download Hasil:")
                    
                    # Buat file CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"batch_prediction_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Tampilkan detail per gambar
                    st.subheader("🔍 Detail Per Gambar:")
                    
                    for idx, result in enumerate(results):
                        with st.expander(f"📷 {result['filename']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Tampilkan gambar
                                uploaded_files[idx].seek(0)  # Reset file pointer
                                st.image(uploaded_files[idx], use_column_width=True)
                            
                            with col2:
                                # Tampilkan hasil prediksi
                                if result['predicted_class'] != 'Error':
                                    st.success(f"**Prediksi:** {result['predicted_class']}")
                                    st.metric("Tingkat Kepercayaan", f"{result['confidence']:.2f}%")
                                    
                                    # Bar chart probabilitas
                                    prob_data = {
                                        'Kelas': CLASSES,
                                        'Probabilitas': [result['normal_prob'], result['aktif_prob'], result['laten_prob']]
                                    }
                                    st.bar_chart(prob_data, x='Kelas', y='Probabilitas', use_container_width=True)
                                else:
                                    st.error(f"Error: {result.get('error', 'Unknown error')}")
# Halaman Tentang
elif st.session_state.current_page == "about":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("---")
    
    st.header("Informasi Aplikasi")
    st.write("""
    Aplikasi Deteksi Chest X-ray ini dikembangkan menggunakan teknologi machine learning 
    untuk membantu dalam identifikasi kondisi pada gambar chest x-ray.
    """)
    
    st.subheader("Teknologi yang Digunakan:")
    st.write("""
    - **Framework**: Streamlit untuk interface web
    - **Machine Learning**: TensorFlow & Keras
    - **Model**: VGG-19 dan Xception
    - **Bahasa Pemrograman**: Python
    """)
    
    st.subheader("Cara Penggunaan:")
    st.write("""
    1. Pilih halaman 'Deteksi' dari sidebar
    2. Pilih model yang ingin digunakan (VGG-19 atau Xception)
    3. Upload gambar chest x-ray (format: PNG, JPEG, JPG)
    4. Klik tombol 'Mulai Prediksi'
    5. Lihat hasil prediksi dan tingkat kepercayaan
    """)
    
    st.subheader("Kelas yang Dapat Diidentifikasi:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Normal**\nKondisi chest x-ray yang normal")
    
    with col2:
        st.warning("**Aktif**\nKondisi tuberculosis aktif")
    
    with col3:
        st.error("**Laten**\nKondisi tuberculosis laten")
    
    st.markdown("---")
    st.caption("© 2024 Aplikasi Deteksi Chest X-ray")

# Footer untuk semua halaman
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit")