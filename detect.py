import cv2

def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), 2, cv2.LINE_AA)
    return frameOpencvDnn, faceBoxes


def detect_faces_in_frame(frame):
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox

        # Safely crop the face with padding
        face = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
                     max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]

        # Skip if face crop is invalid
        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
            continue

        try:
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = f"{gender}, {age}"
            print(f"Detected -> Gender: {gender}, Age: {age}")
            cv2.putText(resultImg, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"[Warning] Face processing failed: {e}")
            continue

    return resultImg


# Load model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-30)',
           '(30-35)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Start webcam
print("[INFO] Starting webcam...")
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    hasFrame, frame = video.read()
    if not hasFrame or frame is None:
        print("[Warning] Frame not captured, skipping...")
        continue

    frame = cv2.resize(frame, (640, 480))
    result = detect_faces_in_frame(frame)
    cv2.imshow("Age and Gender Detection", result)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        print("[INFO] Exiting...")
        break

video.release()
cv2.destroyAllWindows()
