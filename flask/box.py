from flask import Blueprint, render_template, Response
import cv2
import random
import time
import mediapipe as mp

box_bp = Blueprint('box', __name__, template_folder='templates')

circle_speed = 15
score = 0
miss = 0
objects = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(1)  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

print(f"Камера использует разрешение: {frame_width}x{frame_height}")

thug_image = cv2.imread("thug.png", cv2.IMREAD_UNCHANGED)
thug_image = cv2.resize(thug_image, (int(frame_width * 0.1), int(frame_height * 0.1)))  
thug_height, thug_width, _ = thug_image.shape

def add_object():
    x = random.randint(0, frame_width - thug_width)
    y = -thug_height
    objects.append([x, y])

last_object_time = time.time()

def generate_frames():
    global score, miss, objects, last_object_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        if time.time() - last_object_time > 1:
            add_object()
            last_object_time = time.time()

        cv2.putText(frame, f"Score: {score} Miss: {miss}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@box_bp.route('/')
def index():
    return render_template('box.html')

@box_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
