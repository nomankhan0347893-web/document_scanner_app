# 📄 Document Scanner Pro

An AI-powered **document scanning web application** built with **Streamlit + OpenCV** that converts images into clean, high-quality scanned documents using image processing techniques like edge detection and perspective transformation.

---

## 🌐 Live Demo

👉 Try the app here:  
https://document-scan-noman.streamlit.app/

---

## 📌 Project Overview

This project simulates a real-world **scanner system** using computer vision techniques.  
It detects document edges, applies perspective correction, and generates a clean scanned output.

The system transforms any document image into a professional scanned version in seconds.

---

## 🚀 Features

✔ Upload document images (JPG, JPEG, PNG)  
✔ Automatic document edge detection  
✔ Perspective transformation (scan effect)  
✔ Image enhancement & cleanup  
✔ Adjustable cropping tool  
✔ Edge detection preview  
✔ Download scanned output  
✔ Clean dark-mode UI  

---

## 🧠 Tech Stack

- Python 🐍  
- Streamlit 🎈  
- OpenCV 👁️  
- NumPy 🔢  
- Pillow 🖼️  

---

## 📂 Project Structure

Document-Scanner-Pro/
│
├── app.py # Main Streamlit application
├── utils/
│ ├── image_utils.py # Image loading & resizing functions
│ ├── processing_utils.py # Edge detection & scanning logic
│
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ How It Works

1. User uploads a document image  
2. Image is resized for processing  
3. Edge detection identifies document boundaries  
4. Contours are extracted  
5. Perspective transform creates scanned view  
6. Image is cleaned & enhanced  
7. User can crop and download final scan  

---

## 🧪 Processing Pipeline

- Image resizing for optimization  
- Edge detection (Canny algorithm)  
- Contour detection  
- Perspective transformation  
- Image enhancement & cleanup  

---

## 📸 UI Features

- Sidebar with usage guide  
- Live preview of original image  
- Edge detection visualization  
- Interactive cropping sliders  
- One-click download option  

---

## ⚠️ Limitations

- Works best with clear rectangular documents  
- Poor lighting reduces accuracy  
- Curved or folded pages may distort results  
- Complex backgrounds may affect edge detection  

---

## 🔮 Future Improvements

- 📱 Mobile camera scanning support  
- 🤖 AI-based document boundary detection  
- ✍️ Text extraction (OCR integration)  
- 🌍 Multi-language document support  
- ⚡ Faster real-time processing  

---

## 👨‍💻 Developer

**Noman Khan**  
Computer Vision & AI Enthusiast  

- 💻 GitHub: https://github.com/nomankhan0347893-web  
- 🔗 LinkedIn: https://www.linkedin.com/in/noman-khan-95787139b  

---

## 🌐 Deployment

Deployed using **Streamlit Cloud**

👉 https://document-scan-noman.streamlit.app/

---

## 📜 License

This project is open-source and available for educational use.
