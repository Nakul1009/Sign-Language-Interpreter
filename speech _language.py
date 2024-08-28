import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
languages = ['English', 'Tamil', 'Telugu', 'Hindi', 'Malayalam', 
             'Gujarati', 'Marathi', 'Kannada', 'Bengali', 'Punjabi']
selected_language = None
cv2.namedWindow('Language Selection by Sign', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Language Selection by Sign', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    finger_count = [0, 0]
    total_fingers = 0
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[idx].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if handedness == 'Right':
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    finger_count[idx] += 1
            else:
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    finger_count[idx] += 1
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y: 
                finger_count[idx] += 1
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                finger_count[idx] += 1
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                finger_count[idx] += 1
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                finger_count[idx] += 1
        if len(result.multi_hand_landmarks) == 1:
            total_fingers = finger_count[0]
        elif len(result.multi_hand_landmarks) == 2:
            if finger_count[0] == 5 and finger_count[1] == 5:
                total_fingers = 10
            else:
                total_fingers = finger_count[0] + finger_count[1]
    if 1 <= total_fingers <= 10:
        selected_language = languages[total_fingers-1]
    for i, language in enumerate(languages):
        color = (0, 255, 0) if selected_language == language else (255, 255, 255)
        cv2.putText(frame, f'{i + 1}. {language}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if selected_language:
        cv2.putText(frame, f'Selected Language: {selected_language}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Language Selection by Sign', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if selected_language:
    print(f'Selected Language: {selected_language}')
else:
    print("No language was selected.")
