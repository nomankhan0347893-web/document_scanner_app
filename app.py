import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Scanner", layout="wide")

from utils.image_utils import load_image, convert_to_grayscale, resize_image
from utils.processing_utils import (
    fix_lighting, detect_edges, find_contours, order_points,
    perspective_transform, clean_scan, enhance_scan
)


# 🎨 CUSTOM STYLING (ONLY BACKGROUND)
st.markdown("""
<style>

/* FORCE FULL BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
}

/* Text color */
html, body, [class*="css"]  {
    color: #e6e6e6 !important;
}

/* Main content block */
.block-container {
    padding-top: 2rem;
}

/* Titles */
h1, h2, h3 {
    color: #4CC9F0 !important;
}

/* Buttons */
.stButton>button {
    background-color: #2dd4bf !important;
    color: black !important;
    border-radius: 6px;
}

/* Hyperlinks */
a {
    color: #4CC9F0 !important;
}
<style>
a {
    text-decoration: none;
}

a:hover {
    transform: scale(1.1);
    transition: 0.2s;
}



</style>
""", unsafe_allow_html=True)


#  SIDEBAR CONTROLS
st.sidebar.title("⚙️ Settings")

max_dim = st.sidebar.slider("Resize Max Dimension", 400, 1200, 800)

show_edges = st.sidebar.checkbox("Show Edge Detection", True)
show_original = st.sidebar.checkbox("Show Original Image", True)



#  TITLE

st.markdown("<h1> Document Scanner Pro</h1>", unsafe_allow_html=True)
st.write("Upload an image to convert it into a clean scanned document ✨")





# 👇 THEN FILE UPLOAD STARTS
uploaded_file = st.file_uploader(
    "Click the button below to upload an image",
    type=["jpg", "jpeg", "png"]
)

# FILE UPLOAD
st.info("""
💡 **Tips for best results:**
- Use clear document images 📄  
- Avoid dark or blurry photos 🌑  
- Keep document inside frame  
- Use flat surface (no folds)  
- Good lighting improves accuracy 💡  
""")


if uploaded_file is not None:

    progress = st.progress(0)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

  
    #  ORIGINAL IMAGE
    if show_original:
        with col1:
            st.subheader(" Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    progress.progress(20)

    
    #  PROCESSING
    
    with st.spinner(" Plz Wait Scanning document..."):

        resized_image, scale_factor = resize_image(image, max_dim=max_dim)
        progress.progress(40)

        edges = detect_edges(resized_image)
        progress.progress(60)

        contours = find_contours(edges, resized_image)
        progress.progress(75)

        if contours is not None:

            contour = contours * (1 / scale_factor)

            warped = perspective_transform(image, contour)
            cleaned_scan = clean_scan(warped)
            scanned = enhance_scan(cleaned_scan)

            progress.progress(100)

            
            #  CROPPING FEATURE
            with col2:
                st.subheader("Your Scanned Document")

                h, w = scanned.shape[:2]

                st.sidebar.markdown("###  Crop Settings")

                x1 = st.sidebar.slider("Crop X Start", 0, w, 0)
                x2 = st.sidebar.slider("Crop X End", 0, w, w)

                y1 = st.sidebar.slider("Crop Y Start", 0, h, 0)
                y2 = st.sidebar.slider("Crop Y End", 0, h, h)

                if x2 > x1 and y2 > y1:
                    cropped = scanned[y1:y2, x1:x2]
                else:
                    st.warning(" Invalid crop selection, showing full image")
                    cropped = scanned

                st.image(cropped)

           
            #  EDGE VIEW
            if show_edges:
                st.subheader("Edge Detection")
                st.image(edges)

           
            # ⬇️ DOWNLOAD CROPPED IMAGE
            _, buffer = cv2.imencode(".png", cropped)

            st.download_button(
                label="⬇️ Download Scanned Document",
                data=buffer.tobytes(),
                file_name="scanned_document.png",
                mime="image/png"
            )

        else:
            st.error(" Could not detect a document. Try a clearer image.")

# 👇 ADD ABOUT SECTION HERE
st.markdown("## About Developer")

st.markdown("""
<div style="display:flex; gap:20px; align-items:center;">

<a href="https://github.com/nomankhan0347893-web" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40"/>
</a>

<a href="https://www.linkedin.com/in/noman-khan-95787139b/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3Bsw4LIOFOR9aOXnh7K37btA%3D%3D" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40"/>
</a>

</div>
<br>
**Noman Khan**  
<br>
Computer Vision & AI Enthusiast  
Building intelligent document scanning systems ✨
""", unsafe_allow_html=True)