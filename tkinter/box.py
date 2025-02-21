import cv2
import random
import time
import mediapipe as mp
import numpy as np
import argparse
import subprocess
import sys  # Add this import

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=int, default=15)
parser.add_argument("--scale", type=float, default=0.1)
parser.add_argument("--camera", type=int, default=1)
parser.add_argument("--difficulty", type=str, default="easy")
args = parser.parse_args()

circle_speed = args.speed
scale_factor = args.scale
camera_index = args.camera
difficulty = args.difficulty

score = 0
miss = 0
objects = []

if difficulty == "easy":
    lives = 10
elif difficulty == "medium":
    lives = 6
else:  # hard
    lives = 3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(camera_index)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

thug_image = cv2.imread("thug.png", cv2.IMREAD_UNCHANGED)

new_width = int(frame_width * scale_factor)
new_height = int(frame_height * scale_factor)
thug_image = cv2.resize(thug_image, (new_width, new_height))
thug_height, thug_width, _ = thug_image.shape

def add_object():
    global circle_speed
    x = random.randint(0, frame_width - thug_width)
    y = -thug_height
    objects.append([x, y])
    if score + miss != 0 and (score + miss) % 3 == 0:
            circle_speed += 3

last_object_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    hand_rects = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame_width)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame_height)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame_width)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame_height)
            hand_rects.append((x_min, y_min, x_max, y_max))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    for obj in objects[:]:
        obj[1] += circle_speed

        y1, y2 = max(0, obj[1]), min(frame_height, obj[1] + thug_height)
        x1, x2 = max(0, obj[0]), min(frame_width, obj[0] + thug_width)

        y1o, y2o = max(0, -obj[1]), thug_height - max(0, obj[1] + thug_height - frame_height)
        x1o, x2o = max(0, -obj[0]), thug_width - max(0, obj[0] + thug_width - frame_width)

        if (y1 < y2 and x1 < x2) and (y1o < y2o and x1o < x2o):
            thug_resized = cv2.resize(thug_image[y1o:y2o, x1o:x2o], (x2 - x1, y2 - y1))
            alpha_s = thug_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha_s * thug_resized[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c]
                )

        for hand_rect in hand_rects:
            x_min, y_min, x_max, y_max = hand_rect
            if x_min <= obj[0] + thug_width // 2 <= x_max and y_min <= obj[1] + thug_height // 2 <= y_max:
                objects.remove(obj)
                score += 1
                break

        if obj[1] > frame_height:
            objects.remove(obj)
            miss += 1
            lives -= 1

    if lives <= 0:
        cap.release()
        cv2.destroyAllWindows()
        subprocess.Popen([sys.executable, "game_over.py", "--score", str(score)])
        break

    if time.time() - last_object_time > 1:
        add_object()
        last_object_time = time.time()
    
    cv2.putText(frame, f"Score: {score} Lives: {lives}  Speed: {circle_speed}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Box Game', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
