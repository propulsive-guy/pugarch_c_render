import streamlit as st
import requests
import tempfile
import base64
from PIL import Image
import io

st.set_page_config(page_title="Cleanliness Score Dashboard", page_icon="🧼", layout="centered")

st.markdown("""
    <h1 style="text-align:center; color:#88C0D0;">🧼 Cleanliness Score Dashboard</h1>
    <p style="text-align:center; color:#D8DEE9;">Upload an image and get cleanliness level prediction from API</p>
    <hr>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_image_path = tmp.name

    st.image(temp_image_path, caption="📷 Uploaded Image", use_column_width=True)

    api_url = "http://192.168.29.54:8080/predict"  # Change this to your App Engine URL once deployed

    with open(temp_image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(api_url, files=files)

    if response.status_code == 200:
        data = response.json()

        st.markdown(f"### 🧼 Cleanliness Score: **{data['cleanliness_score']} / 10**")
        st.markdown(f"### 💥 Raw Dirtiness Score: **{data['raw_score']}**")

        st.markdown("### 📊 Class-wise Detection Summary")
        for item in data["breakdown"]:
            st.write(f"- **{item['class']}**: {item['count']} × avg conf {item['avg_conf']} × weight {item['weight']}")

        if data.get("image_base64"):
            image_data = base64.b64decode(data["image_base64"])
            st.image(Image.open(io.BytesIO(image_data)), caption="🔍 Detection Output", use_column_width=True)
    else:
        st.error("❌ Failed to get prediction from the API.")
