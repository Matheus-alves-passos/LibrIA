# Importação das bibliotecas necessárias
# Import of required libraries
import cv2  # Para captura e processamento de vídeo / For video capture and processing
import os   # Para operações com sistema de arquivos / For filesystem operations

# Definição do diretório para salvar os dados
# Definition of directory to save data
DATA_DIR = './data'

# Cria o diretório de dados se não existir
# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Configuração dos parâmetros do dataset
# Dataset parameters configuration
number_of_classes = 5     # Número de classes a serem coletadas / Number of classes to collect
dataset_size = 100        # Quantidade de imagens por classe / Number of images per class

# Inicializa a captura de vídeo da webcam
# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Loop através de cada classe
# Loop through each class
for j in range(number_of_classes):
    # Cria um diretório para cada classe
    # Create a directory for each class
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    done = False
    
    # Aguarda o usuário pressionar 'q' para iniciar a captura
    # Wait for user to press 'q' to start capture
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Captura e salva as imagens para a classe atual
    # Capture and save images for current class
    counter = 0
    while counter < dataset_size:
        # Captura um frame da webcam
        # Capture a frame from webcam
        ret, frame = cap.read()
        
        # Mostra o frame atual
        # Show current frame
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Salva o frame como imagem
        # Save frame as image
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

# Libera os recursos e fecha as janelas
# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
