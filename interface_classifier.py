import mediapipe as mp  # PT: Biblioteca para análise de gestos das mãos | EN: Library for hand gesture analysis
import cv2  # PT: Biblioteca para captura de vídeo da câmera | EN: Library for camera video capture
import numpy as np  # PT: Biblioteca para cálculos matemáticos e análise de dados | EN: Library for mathematical calculations and data analysis
import pickle  # PT: Biblioteca para salvar e carregar dados | EN: Library for saving and loading data
import pyttsx3  # PT: Biblioteca para síntese de voz | EN: Library for text-to-speech synthesis
import time  # PT: Biblioteca para controle de tempo | EN: Library for time control
from threading import Thread  # PT: Para executar processos em paralelo | EN: For parallel processing
from queue import Queue  # PT: Para gerenciar fila de sinais detectados | EN: For managing detected signs queue

# PT: Inicialização do motor de voz | EN: Initialize text-to-speech engine
engine = pyttsx3.init()

# PT: Configuração da voz em português | EN: Portuguese voice configuration
voices = engine.getProperty('voices')
portuguese_voice = None
for voice in voices:
    if "portuguese" in voice.name.lower() or "brazil" in voice.name.lower():
        portuguese_voice = voice
        break

# PT: Define a voz em português ou usa a padrão | EN: Set Portuguese voice or use default
if portuguese_voice:
    engine.setProperty('voice', portuguese_voice.id)
else:
    print("Aviso: Voz em português não encontrada. Usando voz padrão.")
    # EN: Warning: Portuguese voice not found. Using default voice.

# PT: Configurações de otimização da voz | EN: Voice optimization settings
engine.setProperty('rate', 200)      # PT: Velocidade da fala | EN: Speech rate
engine.setProperty('volume', 1)      # PT: Volume | EN: Volume
engine.setProperty('pitch', 150)     # PT: Tom da voz | EN: Voice pitch

# PT: Inicialização das variáveis de controle da fila de fala | EN: Initialize speech queue control variables
speech_queue = Queue()
last_spoken_time = time.time()
last_spoken_text = ""
speech_thread_running = True

def speak_worker():
    """
    PT: Função que processa a fila de texto para fala em uma thread separada
    EN: Function that processes the text-to-speech queue in a separate thread
    """
    global speech_thread_running
    while speech_thread_running:
        try:
            text = speech_queue.get(timeout=1)
            text_with_pauses = " ... ".join(text.split())
            engine.say(text_with_pauses)
            engine.runAndWait()
            speech_queue.task_done()
        except:
            continue

def speak_text(text):
    """
    PT: Adiciona texto à fila de fala com controle de tempo
    EN: Adds text to the speech queue with time control
    """
    global last_spoken_time, last_spoken_text
    current_time = time.time()
    
    if (current_time - last_spoken_time >= 2.5 and text != last_spoken_text):
        speech_queue.put(text)
        last_spoken_time = current_time
        last_spoken_text = text

# PT: Inicia a thread de fala | EN: Start speech thread
speech_thread = Thread(target=speak_worker, daemon=True)
speech_thread.start()

# PT: Carrega o modelo treinado | EN: Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# PT: Inicializa a captura de vídeo | EN: Initialize video capture
cap = cv2.VideoCapture(0)

# PT: Configuração da janela em tela cheia | EN: Fullscreen window configuration
cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# PT: Configuração do MediaPipe para detecção de mãos | EN: MediaPipe hand detection setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

# PT: Dicionário de rótulos para os sinais | EN: Label dictionary for signs
labels_dict = {0: 'Oi', 1: 'Meu', 2: 'Nome', 3: 'Matheus', 4: 'Faz o L'}

def format_speech_text(detected_signs):
    """
    PT: Formata o texto para fala baseado nos sinais detectados
    EN: Formats speech text based on detected signs
    """
    if len(detected_signs) == 1:
        return f"{detected_signs[0]}"
    elif len(detected_signs) == 2:
        return f"{detected_signs[0]}{detected_signs[1]}"
    elif len(detected_signs) > 2:
        signs_text = "".join(detected_signs[:-1]) + f"{detected_signs[-1]}"
        return f"{signs_text}"
    return ""

try:
    # PT: Loop principal de processamento de vídeo | EN: Main video processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # PT: Processamento da imagem | EN: Image processing
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        detected_signs = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # PT: Desenha os pontos de referência das mãos | EN: Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # PT: Prepara os dados para análise | EN: Prepare data for analysis
                data_aux = []
                x_ = []
                y_ = []

                # PT: Coleta as coordenadas dos pontos | EN: Collect landmark coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # PT: Normaliza as coordenadas | EN: Normalize coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # PT: Calcula as coordenadas do retângulo | EN: Calculate rectangle coordinates
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    # PT: Realiza a predição do sinal | EN: Perform sign prediction
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]
                    detected_signs.append(predicted_character)

                    # PT: Desenha o retângulo e o texto | EN: Draw rectangle and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                               cv2.LINE_AA)
                except Exception as e:
                    print(f"Erro de previsão para mão {hand_idx}: {str(e)}")
                    continue

            # PT: Processa os sinais detectados | EN: Process detected signs
            if detected_signs:
                speech_text = format_speech_text(detected_signs)
                speak_text(speech_text)

        # PT: Exibe o frame e verifica tecla de saída | EN: Display frame and check exit key
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # PT: Finaliza todos os recursos | EN: Clean up all resources
    speech_thread_running = False
    speech_thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
