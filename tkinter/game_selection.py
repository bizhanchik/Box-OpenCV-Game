import pickle
import cv2
import sys
import mediapipe as mp
import numpy as np
import subprocess
import time

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

labels_dict = {0: 'Left', 1: 'Right', 2: 'One', 3: 'Two', 4: 'Three'}

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(camera_index)

selected_game = None
gesture_start_time = None
detected_gesture = None
HOLD_TIME = 2 

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    cv2.putText(frame, "Right -> Box | Left -> Snake", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                gesture = labels_dict[int(prediction[0])]

                if gesture != detected_gesture:
                    detected_gesture = gesture
                    gesture_start_time = time.time()
                else:
                    elapsed_time = time.time() - gesture_start_time
                    cv2.putText(frame, f"{gesture} ({elapsed_time:.1f}s)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if elapsed_time >= HOLD_TIME:
                        selected_game = "box" if gesture == "Right" else "snake"
                        break

    cv2.imshow("Выбор игры", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if selected_game is not None:
        break

cap.release()
cv2.destroyAllWindows()

if selected_game == "box":
    subprocess.Popen([sys.executable, "box_dif.py", str(camera_index)])
elif selected_game == "snake":
    print("Игра Snake не реализована.")
