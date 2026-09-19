import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Cấu hình trang
st.set_page_config(
    page_title="N&N Speckle Contrast", page_icon="🔬", layout="centered"
)

st.title("🔬 N&N SPECKLE CONTRAST")
st.write("Ứng dụng phân tích Speckle Contrast trực tiếp trên trình duyệt")

# ----------------------------------------------------
# TAB 1: PHÂN TÍCH ẢNH (CAMERA / TẢI ẢNH)
# ----------------------------------------------------
st.header("1. Phân tích ảnh Speckle")

input_type = st.radio(
    "Lựa chọn nguồn ảnh:",
    ("📷 Chụp ảnh trực tiếp", "📁 Tải ảnh từ thiết bị"),
    horizontal=True,
)

img_bgr = None

if input_type == "📷 Chụp ảnh trực tiếp":
    camera_file = st.camera_input("Chụp ảnh mẫu Speckle")
    if camera_file is not None:
        file_bytes = np.asarray(
            bytearray(camera_file.read()), dtype=np.uint8
        )
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    uploaded_file = st.file_uploader(
        "Chọn tệp ảnh mẫu (.jpg, .png, .jpeg)",
        type=["jpg", "png", "jpeg"],
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_bgr is not None:
    st.image(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        caption="Ảnh đầu vào",
        use_column_width=True,
    )

    if st.button("⚡ TÍNH SPECKLE CONTRAST", type="primary"):
        # Chuyển ảnh xám
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Tính mean và variance theo cửa sổ lọc (5x5)
        mean = cv2.blur(gray, (5, 5))
        mean_sq = cv2.blur(gray**2, (5, 5))
        variance = np.maximum(mean_sq - mean**2, 0)

        # Tính contrast
        contrast = np.divide(
            np.sqrt(variance), mean, out=np.zeros_like(gray), where=mean != 0
        )

        k_val = float(np.mean(contrast))
        q_val = (1.0 / (k_val**2)) if k_val != 0 else 0.0

        # Hiển thị kết quả dạng chỉ số
        col1, col2 = st.columns(2)
        col1.metric("Chỉ số Speckle (K)", f"{k_val:.4f}")
        col2.metric("Lưu lượng Q (1/K²)", f"{q_val:.2f}")

        # Tạo bản đồ Heatmap
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

st.divider()

# ----------------------------------------------------
# TAB 2: VẼ ĐỒ THỊ TASK 3 & TASK 5
# ----------------------------------------------------
st.header("2. Vẽ đồ thị Task 3 & Task 5")

k_input = st.text_input(
    "Nhập chuỗi giá trị K (phân cách bởi dấu phẩy):",
    "0.15, 0.12, 0.18, 0.09, 0.14",
)

col_t3, col_t5 = st.columns(2)

with col_t3:
    if st.button("📈 Vẽ Task 3 (Q - t)"):
        try:
            k_list = [
                float(x.strip()) for x in k_input.split(",") if x.strip()
            ]
            q_list = [(1.0 / (k**2)) for k in k_list]
            t_list = list(range(1, len(k_list) + 1))

            fig, ax = plt.subplots()
            ax.plot(t_list, q_list, "r-s", linewidth=2, label="Q = 1/K²")
            ax.set_xlabel("Thời gian (t)")
            ax.set_ylabel("Lưu lượng (Q)")
            ax.set_title("Đồ thị Q = 1/K² theo t")
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Lỗi nhập liệu: {e}")

with col_t5:
    if st.button("📊 Vẽ Task 5 (K - Q)"):
        try:
            k_list = [
                float(x.strip()) for x in k_input.split(",") if x.strip()
            ]
            q_list = [(1.0 / (k**2)) for k in k_list]

            fig, ax = plt.subplots()
            ax.plot(k_list, q_list, "g-^", linewidth=2, label="K - Q")
            ax.set_xlabel("Chỉ số K")
            ax.set_ylabel("Lưu lượng (Q)")
            ax.set_title("Đồ thị K - Q")
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Lỗi nhập liệu: {e}")
