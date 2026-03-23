import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 224
model = tf.keras.models.load_model("hand_sign_model.keras")

labels = sorted(os.listdir("dataset"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            h, w, _ = frame.shape
            xs, ys = [], []

            for lm in handLms.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            xmin, xmax = max(0,min(xs)-20), min(w,max(xs)+20)
            ymin, ymax = max(0,min(ys)-20), min(h,max(ys)+20)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size:
                input_img = preprocess(hand_img)
                preds = model.predict(input_img, verbose=0)[0]

                idx = np.argmax(preds)
                label = labels[idx].replace("-samples","")
                confidence = preds[idx]*100

                cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                cv2.putText(frame,f"{label} {confidence:.2f}%",
                            (xmin,ymin-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()