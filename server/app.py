from flask import Flask, jsonify, Response, request
import cv2
import mediapipe as mp
import threading
import queue
import pyttsx3
from flask_cors import CORS
import random
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 150)  # Adjust speech speed
except Exception as e:
    print("Error initializing TTS engine:", e)
    exit()

# Queue for TTS requests
tts_queue = queue.Queue()

# Global variables for hand detection
current_fingers = 0
current_hand = ""
frame_lock = threading.Lock()
stabilization_delay = 0.5  # Stabilization delay in seconds
last_fingers = -1
last_hand = ""
last_change_time = 0

# Game state variables
game_state = {
    "active": False,
    "waiting_for_response": False,
    "current_question": "",
    "correct_answer": 0,
    "question_start_time": 0,
    "tries_left": 2,
    "score": 0,
    "questions_asked": 0,
    "total_questions": 10,
    "processing_answer": False,
    "question_cooldown": False,
    "remaining_questions": []
}
game_lock = threading.Lock()

# Question bank
ORIGINAL_QUESTIONS = [
    ("How many eyes do you have?", 2),
    ("How many fingers do you have on one hand?", 5),
    ("How many legs do you have?", 2),
    ("How many thumbs do you have?", 2),
    ("How many noses do you have?", 1),
    ("How many ears do you have?", 2),
    ("How many hands do you have?", 2),
    ("How many feet do you have?", 2),
    ("How many wheels does a car have?", 4),
    ("How many legs does a chair have?", 4),
    ("How many sides does a triangle have?", 3),
    ("How many colors are in a rainbow?", 7),
    ("How many days are in a week?", 7),
    ("How many letters are in the word 'cat'?", 3),
    ("How many legs does a dog have?", 4)
]

# Game state with thread locks
game_state = {
    "active": False,
    "waiting_for_response": False,
    "current_question": "",
    "correct_answer": 0,
    "question_start_time": 0,
    "tries_left": 2,
    "score": 0,
    "questions_asked": 0,
    "total_questions": 10,
    "processing_answer": False,
    "remaining_questions": []
}
game_lock = threading.Lock()
frame_lock = threading.Lock()

# Hand detection variables
current_fingers = 0
current_hand = ""
last_fingers = -1
last_hand = ""
last_change_time = 0
stabilization_delay = 1.0

def count_fingers(hand_landmarks, hand_label):
    fingers = 0
    finger_tips = [8, 12, 16, 20]
    
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers += 1

    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]

    if hand_label == "Right":
        if thumb_tip.x < thumb_mcp.x:
            fingers += 1
    else:
        if thumb_tip.x > thumb_mcp.x:
            fingers += 1

    return fingers

def ask_question():
    with game_lock:
        if not game_state["active"]:
            return False

        if len(game_state["remaining_questions"]) == 0:
            game_state["remaining_questions"] = ORIGINAL_QUESTIONS.copy()
            random.shuffle(game_state["remaining_questions"])

        if not game_state["remaining_questions"]:
            return False

        question, answer = game_state["remaining_questions"].pop(0)
        game_state.update({
            "current_question": question,
            "correct_answer": answer,
            "waiting_for_response": True,
            "question_start_time": time.time(),
            "tries_left": 2,
            "processing_answer": False
        })
        
        speak(question)
        return True

def check_answer(fingers):
    with game_lock:
        if not game_state["waiting_for_response"] or game_state["processing_answer"]:
            return
            
        elapsed_time = time.time() - game_state["question_start_time"]
        if elapsed_time < 2:
            return
            
        game_state["processing_answer"] = True
        
        if fingers == game_state["correct_answer"]:
            game_state["score"] += 1
            game_state["questions_asked"] += 1  # Moved here from ask_question()
            speak(f"Correct! Your score is {game_state['score']}")
        else:
            game_state["tries_left"] -= 1
            if game_state["tries_left"] > 0:
                speak(f"Try again. {game_state['tries_left']} attempts left")
                # Reset detection
                global last_fingers, last_hand
                last_fingers = -1
                last_hand = ""
                game_state["question_start_time"] = time.time()
            else:
                game_state["questions_asked"] += 1  # Count attempt after last try
                speak(f"The answer was {game_state['correct_answer']}")
        
        game_state["waiting_for_response"] = False
        game_state["processing_answer"] = False

def generate_frames():
    cap = cv2.VideoCapture(0)
    global current_fingers, current_hand, last_fingers, last_hand, last_change_time

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        fingers = 0
        hand_label = ""

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                fingers += count_fingers(hand_landmarks, hand_label)

        # Stabilization logic
        if fingers != last_fingers or hand_label != last_hand:
            last_change_time = time.time()
            last_fingers = fingers
            last_hand = hand_label

        if time.time() - last_change_time >= stabilization_delay:
            if fingers != current_fingers or hand_label != current_hand:
                if hand_label and game_state["waiting_for_response"]:
                    speak(f"{fingers}")
                    check_answer(fingers)
                with frame_lock:
                    current_fingers = fingers
                    current_hand = hand_label

        # Display game info
        if game_state["active"]:
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            cv2.putText(frame, f"Score: {game_state['score']}", (10, 30), font, 0.7, color, 1)
            cv2.putText(frame, f"Q: {game_state['questions_asked']}/{game_state['total_questions']}", 
                        (10, 60), font, 0.7, color, 1)
            if game_state["current_question"]:
                y_pos = 90
                for line in wrap_text(game_state["current_question"], 50):
                    cv2.putText(frame, line, (10, y_pos), font, 0.5, color, 1)
                    y_pos += 25

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_game', methods=['POST'])
def start_game():
    with game_lock:
        if game_state["active"]:
            return jsonify({"status": "error", "message": "Game already running"})
        
        game_state.update({
            "active": True,
            "score": 0,
            "questions_asked": 0,
            "waiting_for_response": False,
            "remaining_questions": ORIGINAL_QUESTIONS.copy()
        })
        
        random.shuffle(game_state["remaining_questions"])
        ask_question()
        
        return jsonify({
            "status": "success",
            "message": "Game started",
            "current_question": game_state["current_question"],
            "score": 0,
            "questions_asked": 0
        })

@app.route('/next_question', methods=['POST'])
def next_question():
    with game_lock:
        if not game_state["active"]:
            return jsonify({"status": "error", "message": "Game not active"})
            
        if game_state["questions_asked"] >= game_state["total_questions"]:
            game_state["active"] = False
            return jsonify({
                "status": "success",
                "message": f"Game over! Final score: {game_state['score']}",
                "game_over": True
            })
            
        if not ask_question():
            return jsonify({
                "status": "error",
                "message": "Failed to load question"
            })
            
        return jsonify({
            "status": "success",
            "current_question": game_state["current_question"],
            "questions_asked": game_state["questions_asked"],
            "score": game_state["score"]
        })

@app.route('/game_status')
def get_game_status():
    with game_lock:
        return jsonify(game_state)
if __name__ == '__main__':
    app.run(debug=True, port=5000)