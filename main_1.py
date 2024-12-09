import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import cv2
from twilio.rest import Client

def sendAlert(msg):
    print(msg)
    account_sid = 'AC3f877ea5c5f772bf78452998685a1928'
    auth_token = 'a1bcfc507934a6924dd82dde838ebca3'
    # client = Client(account_sid, auth_token)
    # message = client.messages.create(
    # from_='+12513086349',
    # body=msg,
    # to='+919600071484'
    # )
    # print(message.sid)

model1 = load_model('./mobilenetv2_model.h5')
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("./sample_video_fast.mp4")
collision_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame)
    
    for result in results:
        boxes = result.boxes.xyxy 
        scores = result.boxes.conf
        class_ids = result.boxes.cls

        for i in range(len(boxes)):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = map(int, boxes[i])
                cropped_img = frame[y1:y2, x1:x2]
                
                img = cv2.resize(cropped_img, (256, 256))
                img_array = np.expand_dims(img, axis=0) / 255.0

                prediction = model1.predict(img_array)
                threshold = 0.6
                class_label = 1 if prediction[0] > threshold else 0
                print(prediction[0])

                label = f'Predicted class: {class_label} (Prob: {prediction[0][0]:.2f})'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 100), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 2)
                if (class_label):
                    cv2.putText(frame, "COLLISION !!", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if (collision_detected==False):
                        sendAlert("Alert!! Collision Detected at OMR Chennai. Impact of Collision : Severe")
                        collision_detected = True
                
    cv2.imshow('YOLOv8 Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()