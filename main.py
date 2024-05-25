import streamlit as st
from st_pages import Page, show_pages

st.set_page_config(
    page_title="Welcome",
    page_icon="🏠",
    layout="wide"
)

# Add logo and title to the sidebar
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.markdown("<h2 style='text-align: left;'>21110340 - Trình Học Tuấn</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: left;'>21110322 - Đặng Hoàng Toàn</h2>", unsafe_allow_html=True)

show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("pages/face_detection.py", "Nhận diện khuôn mặt", "🙂"),
        Page("pages/fruits_detection.py", "Nhận diện trái cây", "🍏"),
        Page("pages/number_detection.py", "Nhận diện chữ số viết tay mnist", "🔢"),
        Page("pages/Xu_Ly_Anh.py", "Xử lý ảnh", "🖼"),
        Page("pages/Dem_so_ngon_tay.py", "Đếm số ngón tay", "✌️")
    ]
)
st.write("# DIPR430685_23_2_04CLC - Xử lý ảnh👋")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    # Thành viên nhóm:
    ### Trình Học Tuấn - 21110340
    ### Đặng Hoàng Toàn - 21110322
    """
)

# st.rerun()