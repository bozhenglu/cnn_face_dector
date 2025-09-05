import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------
# 載入你訓練好的模型
# -------------------------
model = load_model("my_model.h5")

# 類別對應（改成你的實際類別名稱）
class_labels = {0: "boz", 1: "ron"}

# -------------------------
# 找 Iriun 攝影機
# -------------------------
def find_cam(max_index=5):
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return -1

cam_index = find_cam()
if cam_index == -1:
    raise RuntimeError("找不到攝影機")
print("使用攝影機編號:", cam_index)

# -------------------------
# 開啟攝影機
# -------------------------
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# -------------------------
# 即時預測
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
while True:
    ok, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if not ok:
        print("讀取失敗，可能被其他程式占用或權限問題")
        break

 
    img = cv2.resize(frame, (128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    label = class_labels[pred_class]

    
    cv2.putText(frame, f"hello, {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Iriun (iPhone) - Real-time Prediction", frame)

    if cv2.waitKey(1) == 27:  # ESC 離開
        break

cap.release()
cv2.destroyAllWindows()
