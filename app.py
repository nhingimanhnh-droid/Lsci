import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 1. Cấu hình trang & Giao diện (Dark Theme / Word Report Style)
st.set_page_config(
    page_title="N&N Corp - LASCA Analysis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS để làm đẹp giao diện giống khung báo cáo/tài liệu cao cấp
st.markdown(
    """
    <style>
    /* Nền tổng thể Dark Theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    /* Khung viền style Báo cáo / Document Card */
    .report-card {
        border: 2px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        background-color: #161b22;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Header N&N Corporation */
    .brand-header {
        text-align: center;
        border-bottom: 2px solid #e63946;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .brand-title {
        color: #e63946;
        font-size: 26px;
        font-weight: bold;
        letter-spacing: 1.5px;
        margin: 0;
    }
    .brand-slogan {
        color: #8b949e;
        font-size: 13px;
        font-style: italic;
        margin-top: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# 2. Header Thương hiệu N and N Corporation
st.markdown(
    """
    <div class="brand-header">
        <div class="brand-title">N and N CORPORATION (N²)</div>
        <div class="brand-slogan">"No sacrifice is in vain"</div>
    </div>
""",
    unsafe_allow_html=True,
)

st.title("🔬 LASCA Analysis System")
st.caption(
    "Hệ thống phân tích Laser Speckle Contrast Analysis trực tuyến dành cho thiết bị di động."
)

# ----------------------------------------------------
# TAB 1: PHÂN TÍCH ẢNH SPECKLE
# ----------------------------------------------------
st.markdown('<div class="report-card">', unsafe_allow_html=True)
st.subheader("1. Thu thập & Phân tích mẫu Speckle")

input_type = st.radio(
    "Nguồn dữ liệu đầu vào:",
    ("📷 Chụp ảnh trực tiếp", "📁 Tải tệp từ thiết bị"),
    horizontal=True,
)

img_bgr = None

if input_type == "📷 Chụp ảnh trực tiếp":
    camera_file = st.camera_input("Chụp mẫu Laser Speckle")
    if camera_file is not None:
        file_bytes = np.asarray(
            bytearray(camera_file.read()), dtype=np.uint8
        )
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    uploaded_file = st.file_uploader(
        "Chọn tệp ảnh (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is not None:
    st.image(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        caption="Ảnh mẫu Speckle đầu vào",
        use_column_width=True,
    )

    if st.button("⚡ TÍNH CHỈ SỐ LASCA", type="primary"):
        # Chuyển ảnh xám & Tính Contrast K
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(gray, (5, 5))
        mean_sq = cv2.blur(gray**2, (5, 5))
        variance = np.maximum(mean_sq - mean**2, 0)
        contrast = np.divide(
            np.sqrt(variance), mean, out=np.zeros_like(gray), where=mean != 0
        )

        k_val = float(np.mean(contrast))
        q_val = (1.0 / (k_val**2)) if k_val != 0 else 0.0

        # Hiển thị số liệu kết quả
        col1, col2 = st.columns(2)
        col1.metric("Chỉ số Speckle Contrast (K)", f"{k_val:.4f}")
        col2.metric("Lưu lượng dòng chảy Q (1/K²)", f"{q_val:.2f}")

        # Bản đồ Heatmap
        contrast_norm = cv2.normalize(
            contrast, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        heatmap = cv2.applyColorMap(contrast_norm, cv2.COLORMAP_JET)

        st.subheader("🔥 Bản đồ phân bố dòng chảy (Heatmap)")
        st.image(
            cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            caption=f"Visualized Heatmap (K = {k_val:.4f})",
            use_column_width=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: ĐỒ THỊ BÁO CÁO (TASK 3 & TASK 5)
# ----------------------------------------------------
st.markdown('<div class="report-card">', unsafe_allow_html=True)
st.subheader("2. Đồ thị phân tích Task 3 & Task 5")

k_input = st.text_input(
    "Nhập chuỗi K (cách nhau bằng dấu phẩy):", "0.15, 0.12, 0.18, 0.09, 0.14"
)

col_t3, col_t5 = st.columns(2)

# Cấu hình Matplotlib Dark Mode để hợp với giao diện
plt.style.use("dark_background")

with col_t3:
    if st.button("📈 Đồ thị Task 3 (Q - t)"):
        try:
            k_list = [
                float(x.strip()) for x in k_input.split(",") if x.strip()
            ]
            q_list = [(1.0 / (k**2)) for k in k_list]
            t_list = list(range(1, len(k_list) + 1))

            fig, ax = plt.subplots()
            ax.plot(
                t_list,
                q_list,
                color="#e63946",
                marker="s",
                linewidth=2,
                label="Q = 1/K²",
            )
            ax.set_xlabel("Thời gian (t)")
            ax.set_ylabel("Lưu lượng (Q)")
            ax.set_title("Biến thiên Q theo t")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error("Lỗi định dạng dữ liệu!")

with col_t5:
    if st.button("📊 Đồ thị Task 5 (K - Q)"):
        try:
            k_list = [
                float(x.strip()) for x in k_input.split(",") if x.strip()
            ]
            q_list = [(1.0 / (k**2)) for k in k_list]

            fig, ax = plt.subplots()
            ax.plot(
                k_list,
                q_list,
                color="#2a9d8f",
                marker="^",
                linewidth=2,
                label="K - Q",
            )
            ax.set_xlabel("Chỉ số K")
            ax.set_ylabel("Lưu lượng (Q)")
            ax.set_title("Mối quan hệ K - Q")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error("Lỗi định dạng dữ liệu!")

st.markdown("</div>", unsafe_allow_html=True)
