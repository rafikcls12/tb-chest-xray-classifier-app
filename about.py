import streamlit as st

if st.session_state.current_page == "about":
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