import streamlit as st
from PIL import Image
import numpy as np
from module.nhan_dien_trai_cay import trai_cay_gui as frd

# Cấu hình trang Streamlit
st.set_page_config(page_title="Nhận diện trái cây", page_icon="😃", layout="wide")

st.markdown("# Nhận diện trái cây")
# Nút Open Image để chọn ảnh và hiển thị bản xem trước
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "tif"])
process_button = st.button("Xử lý")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Sử dụng cột để hiển thị ảnh gốc và ảnh đã xử lý trên cùng một hàng
    col1, col2 = st.columns(2)
    col1.image(img_array, caption='Ảnh gốc', width=450)

    if process_button:
        processed_img = frd.detect(img_array)
        
        if processed_img is not None:
            col2.image(processed_img, caption='Ảnh đã nhận diện', width=450)
else:
    if process_button:
        st.warning("Vui lòng chọn một ảnh trước khi nhấn 'Xử lý'.")
