import streamlit as st
import time
import datetime
import numpy as np
from PIL import Image
from preprocessing import preprocess_image, resize_image_for_preview

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