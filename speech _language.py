import cv2
import mediapipe as mp

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Language options for 10 languages
languages = ['English', 'Tamil', 'Telugu', 'Hindi', 'Malayalam', 
             'Gujarati', 'Marathi', 'Kannada', 'Bengali', 'Punjabi']

# Variable to store the selected language
selected_language = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    # Initialize the finger count for both hands
    finger_count = [0, 0]
    total_fingers = 0

    # Check if any hands are detected
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check how many fingers are up for the current hand
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:  # Thumb
                finger_count[idx] += 1
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:  # Index finger
                finger_count[idx] += 1
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:  # Middle finger
                finger_count[idx] += 1
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:  # Ring finger
                finger_count[idx] += 1
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:  # Little finger
                finger_count[idx] += 1

        # Calculate the total finger count based on single or both hands
        if len(result.multi_hand_landmarks) == 1:
            total_fingers = finger_count[0]
        elif len(result.multi_hand_landmarks) == 2:
            # If both hands are detected, adjust the count for languages 6-10
            total_fingers = 5 + finger_count[1] if finger_count[0] == 5 else finger_count[0] + finger_count[1]

    # Update the selected language based on the number of fingers
    if 1 <= total_fingers <= 10:
        selected_language = languages[total_fingers-0]

    # Display language options and highlight the selected one
    for i, language in enumerate(languages):
        color = (0, 255, 0) if selected_language == language else (255, 255, 255)
        cv2.putText(frame, f'{i + 1}. {language}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the selected language on the screen
    if selected_language:
        cv2.putText(frame, f'Selected Language: {selected_language}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame with the detected hand landmarks
    cv2.imshow('Language Selection by Sign', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the selected language when exiting the program
if selected_language:
    print(f'Selected Language: {selected_language}')
else:
    print("No language was selected.")
