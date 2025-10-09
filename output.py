import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
import threading
from deep_translator import GoogleTranslator as g
from gtts import gTTS
import os
from autocorrect import Speller
import vlc

# Load models and necessary components
ascii_dict = {48: '0', 65: 'A', 66: 'B', 68: 'D', 69: 'E', 
              70: 'F', 71: 'G', 72: 'H', 74: 'J', 75: 'K', 
              77: 'M', 78: 'N', 80: 'P', 81: 'Q', 82: 'R', 
              83: 'S', 84: 'T', 87: 'W', 88: 'X', 89: 'Y', 
              90: 'Z', 49: '1', 50: '2', 51: '3', 52: '4', 53: '5', 
              54: '6', 55: '7', 56: '8', 57: '9', 
              67: 'C', 73: 'I', 76: 'L', 79: 'O', 
              85: 'U', 86: 'V', 97: ' '}
lang_list ={
    'afrikaans': 'af',
    'albanian': 'sq',
    'amharic': 'am',
    'arabic': 'ar',
    'armenian': 'hy',
    'assamese': 'as',
    'aymara': 'ay',
    'azerbaijani': 'az',
    'bambara': 'bm',
    'basque': 'eu',
    'belarusian': 'be',
    'bengali': 'bn',
    'bhojpuri': 'bho',
    'bosnian': 'bs',
    'bulgarian': 'bg',
    'catalan': 'ca',
    'cebuano': 'ceb',
    'chichewa': 'ny',
    'chinese (simplified)': 'zh-CN',
    'chinese (traditional)': 'zh-TW',
    'corsican': 'co',
    'croatian': 'hr',
    'czech': 'cs',
    'danish': 'da',
    'dhivehi': 'dv',
    'dogri': 'doi',
    'dutch': 'nl',
    'english': 'en',
    'esperanto': 'eo',
    'estonian': 'et',
    'ewe': 'ee',
    'filipino': 'tl',
    'finnish': 'fi',
    'french': 'fr',
    'frisian': 'fy',
    'galician': 'gl',
    'georgian': 'ka',
    'german': 'de',
    'greek': 'el',
    'guarani': 'gn',
    'gujarati': 'gu',
    'haitian creole': 'ht',
    'hausa': 'ha',
    'hawaiian': 'haw',
    'hebrew': 'iw',
    'hindi': 'hi',
    'hmong': 'hmn',
    'hungarian': 'hu',
    'icelandic': 'is',
    'igbo': 'ig',
    'ilocano': 'ilo',
    'indonesian': 'id',
    'irish': 'ga',
    'italian': 'it',
    'japanese': 'ja',
    'javanese': 'jw',
    'kannada': 'kn',
    'kazakh': 'kk',
    'khmer': 'km',
    'kinyarwanda': 'rw',
    'konkani': 'gom',
    'korean': 'ko',
    'krio': 'kri',
    'kurdish (kurmanji)': 'ku',
    'kurdish (sorani)': 'ckb',
    'kyrgyz': 'ky',
    'lao': 'lo',
    'latin': 'la',
    'latvian': 'lv',
    'lingala': 'ln',
    'lithuanian': 'lt',
    'luganda': 'lg',
    'luxembourgish': 'lb',
    'macedonian': 'mk',
    'maithili': 'mai',
    'malagasy': 'mg',
    'malay': 'ms',
    'malayalam': 'ml',
    'maltese': 'mt',
    'maori': 'mi',
    'marathi': 'mr',
    'meiteilon (manipuri)': 'mni-Mtei',
    'mizo': 'lus',
    'mongolian': 'mn',
    'myanmar': 'my',
    'nepali': 'ne',
    'norwegian': 'no',
    'odia (oriya)': 'or',
    'oromo': 'om',
    'pashto': 'ps',
    'persian': 'fa',
    'polish': 'pl',
    'portuguese': 'pt',
    'punjabi': 'pa',
    'quechua': 'qu',
    'romanian': 'ro',
    'russian': 'ru',
    'samoan': 'sm',
    'sanskrit': 'sa',
    'scots gaelic': 'gd',
    'sepedi': 'nso',
    'serbian': 'sr',
    'sesotho': 'st',
    'shona': 'sn',
    'sindhi': 'sd',
    'sinhala': 'si',
    'slovak': 'sk',
    'slovenian': 'sl',
    'somali': 'so',
    'spanish': 'es',
    'sundanese': 'su',
    'swahili': 'sw',
    'swedish': 'sv',
    'tajik': 'tg',
    'tamil': 'ta',
    'tatar': 'tt',
    'telugu': 'te',
    'thai': 'th',
    'tigrinya': 'ti',
    'tsonga': 'ts',
    'turkish': 'tr',
    'turkmen': 'tk',
    'twi': 'ak',
    'ukrainian': 'uk',
    'urdu': 'ur',
    'uyghur': 'ug',
    'uzbek': 'uz',
    'vietnamese': 'vi',
    'welsh': 'cy',
    'xhosa': 'xh',
    'yiddish': 'yi',
    'yoruba': 'yo',
    'zulu': 'zu'
}

model_dict_1 = pickle.load(open(r'D:\SIH\final_model_1.p', 'rb'))
model_1 = model_dict_1['model']

model_dict_2 = pickle.load(open(r'D:\SIH\final_model_2.p', 'rb'))
model_2 = model_dict_2['model']

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand tracking model
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2)

# Initialize camera and window
cap = cv2.VideoCapture(0)
cv2.namedWindow('Language Selection by Sign', cv2.WINDOW_NORMAL)

# Initialize variables
control = 0
detected_sign = None
previous_sign = None
start_time = None
string = ""
languages = ['English', 'Tamil', 'Telugu', 'Hindi', 'Malayalam', 
             'Gujarati', 'Marathi', 'Kannada', 'Bengali', 'Punjabi']
selected_language = None
language_selection_time = None
language_selection_threshold = 2  # Time in seconds for automatic language selection

x = 0
file_name = 0
player = None

def play_audio(file):
    player = vlc.MediaPlayer(file)
    player.play()
    while player.get_state() != vlc.State.Ended:
        pass
    player.release()

def play_audio_thread(file):
    global player
    player = vlc.MediaPlayer(file)
    player.play()
    while player.get_state() != vlc.State.Ended:
        pass
    player.release()

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read frame from camera
    _, frame = cap.read()
    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Language selection mode
    if control == 0:
        finger_count = [0, 0]
        total_fingers = 0
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
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
            if len(results.multi_hand_landmarks) == 1:
                total_fingers = finger_count[0]
            elif len(results.multi_hand_landmarks) == 2:
                if finger_count[0] == 5 and finger_count[1] == 5:
                    total_fingers = 10
                else:
                    total_fingers = finger_count[0] + finger_count[1]
        if 1 <= total_fingers <= 10:
            detected_language = languages[total_fingers - 1]
            if detected_language != selected_language:
                language_selection_time = time.time()
                selected_language = detected_language
        for i, language in enumerate(languages):
            if selected_language == language:
                color = (0, 255, 0)  # Green color for the selected language
            else:
                color = (50,50,50)  # Red color for other languages
            cv2.putText(frame, f'{i + 1}. {language}', (10, 30 + i * 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # Display "Selected Language:" in white and the selected language in red
            cv2.putText(frame, 'Selected Language:', (10, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f'{selected_language}', (350, 400), cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255),2)
        
        # Automatic language selection check
        if selected_language and language_selection_time and time.time() - language_selection_time >= language_selection_threshold:
            control = 1
            language_selection_time = None

    elif control == 1:
        if results.multi_hand_landmarks and time.time() - x > 1.5:
            if len(results.multi_hand_landmarks) == 1:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                
                        x_.append(x)
                        y_.append(y)
                
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                try:
                    prediction = model_1.predict([np.asarray(data_aux)])
                    detected_sign = ascii_dict[int(prediction)]
                    if detected_sign == previous_sign:
                        if start_time is None:
                            start_time = time.time()
                        elapsed_time = time.time() - start_time
                        
                        if elapsed_time < 1:
                            red = int(255 * (1 - elapsed_time) / 3)
                            green = int(255 * elapsed_time / 3)
                            box_color = (0, green, red)
                        else:
                            box_color = (0, 255, 0)
                            cv2.putText(frame, "Ok", (x1 + 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                            string += detected_sign
                            previous_sign = None
                            x = time.time()
                    else:
                        previous_sign = detected_sign
                        start_time = None
                        box_color = (0, 0, 255)  # Red box for new sign
                    # Draw the rectangle and the predicted character
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(frame, detected_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                
                except:
                    pass
            elif len(results.multi_hand_landmarks) == 2:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x11 = hand_landmarks.landmark[i].x
                        y11 = hand_landmarks.landmark[i].y
                
                        x_.append(x11)
                        y_.append(y11)
                
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                try:
                    prediction = model_2.predict([np.asarray(data_aux)])
                    detected_sign = ascii_dict[int(prediction)]
                    if detected_sign == previous_sign:
                        if start_time is None:
                            start_time = time.time()
                        elapsed_time = time.time() - start_time
                        
                        if elapsed_time < 1:
                            red = int(255 * (1 - elapsed_time) / 3)
                            green = int(255 * elapsed_time / 3)
                            box_color = (0, green, red)
                        else:
                            box_color = (0, 255, 0)
                            cv2.putText(frame, "Ok", (x1 + 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                            string += detected_sign
                            previous_sign = None
                            x = time.time()
                    else:
                        previous_sign = detected_sign
                        start_time = None
                        box_color = (0, 0, 255)  # Red box for new sign
                    # Draw the rectangle and the predicted character
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(frame, detected_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                
                except:
                    pass

    # Display the frame in the same window
    cv2.rectangle(frame, (0, H - 60), (W, H), (200,200,200), -1)  # White background
    cv2.putText(frame, string, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "HELP (h)", (W - 100,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)
    if control == 2:
        cv2.putText(frame, "press d to BACKSPACE", (W - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, "press p to PLAY", (W - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, "press q to EXIT", (W - 350, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, "press T to SWITCH LANGUAGE", (W - 350, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, "press S to SELECT LANGUAGE", (W - 350, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)


        key1 = cv2.waitKey(1) & 0xFF
        if key1 == ord('q'):
            control = 1
    cv2.imshow('Language Selection by Sign', frame)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if control == 0:
            if selected_language:
                print(f'Selected Language: {selected_language}')
                control = 1
            else:
                print("No language was selected.")
    elif key == ord("q"):
        break
    if key == ord('p'):
        if string == '':
            continue
        selected_language = selected_language.lower()
        m = g(source='en', target=selected_language)
        my_t = m.translate(text=string.lower())
        my_t = my_t.lower()
        audio = gTTS(text=my_t, lang=lang_list[selected_language.lower()], slow=False)
        audio_name = f"audio{file_name}.mp3"
        file_name += 1
        cv2.putText(frame, "Translating...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Language Selection by Sign', frame)
        audio.save(audio_name)
        cv2.putText(frame, "Translating...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Language Selection by Sign', frame)
        cd = os.getcwd()
        dir = os.path.join(cd, audio_name)
        audio_file = dir
        audio_playing = False
        # Avoid playing audio if already playing
        if not audio_playing:
            

            audio_playing = True
            audio_thread = threading.Thread(target=play_audio_thread, args=(audio_file,))
            audio_thread.start()
            if file_name > 2:
                os.remove(f"audio{file_name - 3}.mp3")
        string = ""
    if key == ord("t"):
        control = 0
    if key == ord("d"):
        string = string[:-1]
    if key == ord('h'):
        control = 2    
cap.release()
cv2.destroyAllWindows()
