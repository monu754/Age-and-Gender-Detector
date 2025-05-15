# 🎯 Real-Time Gender and Age Detection Using OpenCV

This project performs real-time **gender and age detection** using a webcam feed with OpenCV's deep learning module (`cv2.dnn`). It uses pre-trained models to detect faces and then predict the gender and age of each detected face.

---

## 📸 How It Works

- Uses a webcam to capture video in real-time.
- Detects faces using a pre-trained OpenCV DNN face detector.
- For each detected face, predicts:
  - **Gender** (Male / Female)
  - **Age group** (e.g. 15–20, 25–30, etc.)
- Displays predictions on screen along with a bounding box.

---

## 🧠 Pre-trained Models Used

- **Face Detection**: `opencv_face_detector.pbtxt`, `opencv_face_detector_uint8.pb`
- **Age Detection**: `age_deploy.prototxt`, `age_net.caffemodel`
- **Gender Detection**: `gender_deploy.prototxt`, `gender_net.caffemodel`

You can download these models from [this repo by spmallick](https://github.com/spmallick/learnopencv/tree/master/AgeGender).

---

## 🚀 Getting Started

### 1. Clone the Repository
    ```bash
    git clone 
