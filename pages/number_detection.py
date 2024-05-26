import streamlit as st
from PIL import Image
import numpy as np
import cv2

from module.nhan_dien_chu_so_viet_tay_mnist.gui import predict_digits

# Cấu hình trang Streamlit
st.set_page_config(page_title="Nhận diện chữ số viết tay", page_icon="😃", layout="wide")

st.markdown("# Nhận diện chữ số viết tay")
# Nút Open Image để chọn ảnh và hiển thị bản xem trước
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "tif"])
process_button = st.button("Xử lý")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_gray = image.convert('L')

    image_array = np.array(img_gray)
   
    col1, col2 = st.columns(2)
    col1.image(image, caption='Ảnh đã tải lên', width=500)
    if image_array is not None:
        if process_button:
            predict_result = predict_digits(image_array)
            if predict_result is not None:
                col2.image(predict_result, caption='Ảnh đã nhận diện', width=450)
            
else:
    if process_button:
        st.warning("Vui lòng chọn một ảnh trước khi nhấn 'Xử lý'.")
