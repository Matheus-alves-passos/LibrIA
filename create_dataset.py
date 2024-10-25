# Importação das bibliotecas necessárias
# Import of required libraries
import pickle  # Para serialização de dados / For data serialization
import os  # Para operações com sistema de arquivos / For filesystem operations
import mediapipe as mp  # Para detecção de mãos / For hand detection
import cv2  # Para processamento de imagens / For image processing
import matplotlib.pyplot as plt  # Para visualização / For visualization

# Inicialização do detector de mãos do MediaPipe
# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Definição do diretório com as imagens
# Definition of the directory containing images
DATA_DIR = './data'

# Inicialização das listas para armazenar dados e rótulos
# Initialize lists to store data and labels
data = []
labels = []

# Loop através de todos os diretórios e imagens
# Loop through all directories and images
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Inicialização de listas auxiliares para coordenadas
        # Initialize auxiliary lists for coordinates
        data_aux = []
        x_ = []
        y_ = []

        # Leitura e conversão da imagem para RGB
        # Read and convert image to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processamento da imagem para detectar mãos
        # Process image to detect hands
        results = hands.process(img_rgb)

        # Se mãos forem detectadas, processa os pontos de referência
        # If hands are detected, process the landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Coleta as coordenadas x e y de todos os pontos
                # Collect x and y coordinates of all points
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normaliza as coordenadas subtraindo o valor mínimo
                # Normalize coordinates by subtracting minimum value
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Adiciona os dados processados e rótulos às listas
            # Add processed data and labels to lists
            data.append(data_aux)
            labels.append(dir_)

# Salva os dados processados em um arquivo pickle
# Save processed data to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
