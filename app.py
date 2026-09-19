import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Cấu hình giao diện chuẩn Desktop / Dark Theme
st.set_page_config(
    page_title="N&N Corp - Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tùy chỉnh CSS để đồng bộ giao diện máy tính cao cấp
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .brand-box {
        text-align: center;
        padding: 15px;
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .brand-title {
        color: #f0f6fc;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 0;
    }
    .brand-sub {
        color: #da3633;
        font-size: 13px;
        font-weight: 600;
        font-style: italic;
        margin-top: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header N and N Corporation
st.markdown(
    """
    <div class="brand-box">
        <div class="brand-title">N AND N CORPORATION (N²)</div>
        <div class="brand-sub">"No sacrifice is in vain"</div>
    </div>
""",
    unsafe_allow_html=True,
)

# Thanh điều hướng sidebar
st.sidebar.title("📌 MENU ĐIỀU HÀNH")
page = st.sidebar.radio(
    "Chọn chức năng:",
    ["📷 Phân tích Speckle", "📊 Đồ thị phân tích (Task 3 & 5)"],
)

# ====================================================
# MỤC 1: PHÂN TÍCH SPECKLE
# ====================================================
if page == "📷 Phân tích Speckle":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📷 Thu thập & Phân tích mẫu Speckle")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        input_type = st.radio(
            "Nguồn dữ liệu:",
            ("📷 Chụp ảnh trực tiếp", "📁 Tải ảnh lên"),
            horizontal=True,
        )

        img_bgr = None
        if input_type == "📷 Chụp ảnh trực tiếp":
            camera_file = st.camera_input("Chụp mẫu")
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

    with col_result:
        if img_bgr is not None:
            st.image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                caption="Ảnh gốc đầu vào",
                use_column_width=True,
            )

            if st.button("⚡ TÍNH CHỈ SỐ SPECKLE", type="primary"):
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(
                    np.float32
                )
                mean = cv2.blur(gray, (5, 5))
                mean_sq = cv2.blur(gray**2, (5, 5))
                variance = np.maximum(mean_sq - mean**2, 0)
                contrast = np.divide(
                    np.sqrt(variance),
                    mean,
                    out=np.zeros_like(gray),
                    where=mean != 0,
                )

                k_val = float(np.mean(contrast))
                q_val = (1.0 / (k_val**2)) if k_val != 0 else 0.0

                m1, m2 = st.columns(2)
                m1.metric("Chỉ số Speckle (K)", f"{k_val:.4f}")
                m2.metric("Lưu lượng Q (1/K²)", f"{q_val:.2f}")

                contrast_norm = cv2.normalize(
                    contrast, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                heatmap = cv2.applyColorMap(contrast_norm, cv2.COLORMAP_JET)

                st.subheader("🔥 Bản đồ Heatmap")
                st.image(
                    cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                    caption=f"Heatmap (K = {k_val:.4f})",
                    use_column_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ====================================================
# MỤC 2: TÁCH RIÊNG - BIỂU ĐỒ (TASK 3 & TASK 5)
# ====================================================
else:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📊 Trung tâm Vẽ Đồ thị Phân tích")

    # Ô nhập liệu dữ liệu đầu vào chung
    k_input = st.text_input(
        "Nhập dãy giá trị K (phân cách bằng dấu phẩy):",
        "0.15, 0.12, 0.18, 0.09, 0.14",
    )

    try:
        k_list = [float(x.strip()) for x in k_input.split(",") if x.strip()]
        q_list = [(1.0 / (k**2)) for k in k_list]
        t_list = list(range(1, len(k_list) + 1))

        # Cấu hình đồ thị Matplotlib kiểu máy tính (Dark Theme)
        plt.style.use("dark_background")

        # Chia ô trống chứa 2 mục đồ thị song song như giao diện PC
        col_task3, col_task5 = st.columns(2)

        # Mục 1: Task 3 (Đồ thị Q - t)
        with col_task3:
            st.markdown("### 1. Đồ thị Task 3 (Lưu lượng Q theo t)")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(
                t_list,
                q_list,
                color="#da3633",
                marker="o",
                linewidth=2,
                label="Q = 1/K²",
            )
            ax1.set_xlabel("Thời gian (t)")
            ax1.set_ylabel("Lưu lượng (Q)")
            ax1.set_title("Biến thiên Q theo thời gian t")
            ax1.grid(True, color="#30363d", linestyle="--", alpha=0.7)
            ax1.legend()
            st.pyplot(fig1)

        # Mục 2: Task 5 (Đồ thị K - Q)
        with col_task5:
            st.markdown("### 2. Đồ thị Task 5 (Mối quan hệ K - Q)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(
                k_list,
                q_list,
                color="#238636",
                marker="s",
                linewidth=2,
                label="K - Q",
            )
            ax2.set_xlabel("Chỉ số K")
            ax2.set_ylabel("Lưu lượng (Q)")
            ax2.set_title("Tương quan giữa K và Q")
            ax2.grid(True, color="#30363d", linestyle="--", alpha=0.7)
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(
            "Vui lòng kiểm tra lại định dạng chuỗi K đã nhập (Ví dụ đúng: 0.15, 0.12, 0.18)."
        )

    st.markdown("</div>", unsafe_allow_html=True)
