from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import requests
from box import box_bp

app = Flask(__name__)
app.register_blueprint(box_bp, url_prefix='/box')

with open('model.p', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model = model_data['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {
    0: 'Hand Open', 
    1: 'Left Like', 
    2: 'Right Like'
}

selected_game = None

@app.route('/')
def index():
    return render_template('index.html', selected_game=selected_game)

@app.route('/start', methods=['POST'])
def start_game():
    global selected_game
    game = request.form.get('game')
    selected_game = game
    if game == 'snake':
        return redirect(url_for('snake'))
    elif game == 'boxing':
        return redirect(url_for('box.index'))
    return redirect(url_for('index'))

@app.route('/snake')
def snake():
    return "Игра Змейка началась!" 

@app.route('/gesture_select', methods=['POST'])
def gesture_select():
    global selected_game
    data = request.get_json()
    gesture = data.get('gesture', '')
    
    if gesture == 'Right Like':
        selected_game = 'boxing'
        return jsonify({'status': 'redirect', 'url': url_for('box.index')})
    elif gesture == 'Left Like':
        selected_game = 'snake'
        return jsonify({'status': 'redirect', 'url': url_for('snake')})
    
    return jsonify({'status': 'success', 'selected_game': selected_game})

@app.route('/selected_game')
def selected_game_route():
    return jsonify({'selected_game': selected_game})

def get_frame():
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        return
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            
            for lm in results.multi_hand_landmarks[0].landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            
            for lm in results.multi_hand_landmarks[0].landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                
                if predicted_character in ['Left Like', 'Right Like']:
                    try:
                        requests.post('http://127.0.0.1:5000/gesture_select', json={'gesture': predicted_character})
                    except:
                        pass
                
                cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    del(camera)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
