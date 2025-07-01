import streamlit as st
import time
import datetime
import numpy as np
from PIL import Image
from preprocessing import preprocess_image, resize_image_for_preview


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