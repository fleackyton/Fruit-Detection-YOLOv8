import io
import os

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Fruit Detection System", page_icon="🍎", layout="wide")

st.title("🍎 Fruit Detection System")
st.subheader("GROUP MEMBERS")
st.write("1. Ndayisabye Thierry — 26RP02821")
st.write("AI-powered fruit detection using YOLOv8n.")
st.divider()

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
SUPPORTED_FRUITS = ["apple", "banana", "orange"]

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

st.sidebar.title("⚙️ Detection Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05,
)
st.sidebar.write("Model: YOLOv8n")
st.sidebar.write("Classes: Apple, Banana, Orange")

if not os.path.isfile(MODEL_PATH):
    st.error("Trained model file 'best.pt' was not found.")
    st.info("Place your trained best.pt file in the same folder as app.py.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error("The YOLO model could not be loaded.")
    st.exception(error)
    st.stop()

st.header("📤 Upload Fruit Image")
uploaded_file = st.file_uploader(
    "Choose an image containing apples, bananas, or oranges",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Invalid image file. Please upload another image.")
        st.stop()

    st.subheader("Original Image")
    st.image(image, caption="Uploaded Fruit Image", use_container_width=True)

    if st.button("🔍 Detect Fruits", type="primary"):
        with st.spinner("Detecting fruits..."):
            try:
                results = model.predict(
                    source=image,
                    conf=confidence,
                    iou=0.45,
                    verbose=False,
                )
            except Exception as error:
                st.error("An error occurred during fruit detection.")
                st.exception(error)
                st.stop()

        result = results[0]
        boxes = result.boxes
        annotated_image = Image.fromarray(result.plot()[:, :, ::-1])

        fruit_counts = {"apple": 0, "banana": 0, "orange": 0}
        detection_data = []

        model_names = model.names
        if isinstance(model_names, list):
            model_names = dict(enumerate(model_names))

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0].item())
                fruit = str(model_names.get(class_id, f"class_{class_id}")).strip().lower()
                score = float(box.conf[0].item())

                if fruit in fruit_counts:
                    fruit_counts[fruit] += 1
                    detection_data.append({
                        "Fruit": fruit.capitalize(),
                        "Confidence (%)": round(score * 100, 2),
                    })

        st.divider()
        st.header("📊 Detection Results")
        st.image(annotated_image, caption="Detected Fruits", use_container_width=True)

        total_fruits = sum(fruit_counts.values())
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Fruits", total_fruits)
        col2.metric("🍎 Apples", fruit_counts["apple"])
        col3.metric("🍌 Bananas", fruit_counts["banana"])
        col4.metric("🍊 Oranges", fruit_counts["orange"])

        st.subheader("Detection Details")
        if detection_data:
            dataframe = pd.DataFrame(detection_data)
            st.dataframe(dataframe, use_container_width=True, hide_index=True)

            csv_data = dataframe.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv_data,
                file_name="fruit_detection_results.csv",
                mime="text/csv",
            )

            image_buffer = io.BytesIO()
            annotated_image.save(image_buffer, format="PNG")
            st.download_button(
                "📥 Download Annotated Image",
                data=image_buffer.getvalue(),
                file_name="fruit_detection_result.png",
                mime="image/png",
            )
        else:
            st.warning(
                "No supported fruits were detected. Try lowering the confidence "
                "threshold or check your trained model classes."
            )

st.divider()
st.caption("Fruit Detection System | Powered by YOLOv8n and Streamlit")
