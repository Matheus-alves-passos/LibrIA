# 🖐️ Classificador de Sinais com Mãos (MediaPipe + Random Forest)

Este projeto tem como objetivo capturar sinais manuais por meio da webcam, treinar um modelo de **machine learning** para reconhecê-los e, por fim, usar esse modelo em tempo real com feedback por **síntese de voz**.

---

## 🗂️ Estrutura do Projeto

```
.
├── collect_img.py         # Captura imagens da webcam e salva em pastas
├── create_dataset.py      # Processa imagens salvas e cria um dataset .pickle
├── train_classifier.py    # Treina modelo RandomForest com os dados coletados
├── interface_classifier.py# Interface final: previsão em tempo real + fala
├── data/                  # Diretório gerado contendo subpastas por classe (0, 1, 2, ...)
├── data.pickle            # Arquivo de dataset gerado com dados + rótulos
├── model.p                # Modelo treinado salvo
```

---

## 📦 Dependências

Instale os pacotes necessários com:

```bash
pip install opencv-python mediapipe scikit-learn numpy pyttsx3 matplotlib
```

---

## 📸 Etapa 1: Coleta de Dados (`collect_img.py`)

- Usa a **webcam** para capturar imagens.
- Para cada classe (gesto), cria uma pasta separada.
- Aguarda o usuário apertar `Q` para iniciar a coleta.
- Captura 100 imagens por classe.

> 📁 Resultado: pasta `./data/` contendo subpastas `0`, `1`, `2`, ... com imagens `.jpg`.

---

## 🧠 Etapa 2: Criação do Dataset (`create_dataset.py`)

- Lê todas as imagens salvas.
- Usa o **MediaPipe** para detectar mãos nas imagens.
- Extrai coordenadas normalizadas dos pontos da mão.
- Associa cada conjunto de coordenadas ao rótulo da classe correspondente.

> 💾 Resultado: `data.pickle` com as listas `data` (features) e `labels` (classes).

---

## 🏋️ Etapa 3: Treinamento do Modelo (`train_classifier.py`)

- Carrega o `data.pickle`.
- Divide em treino (80%) e teste (20%).
- Usa **RandomForestClassifier** para treinar.
- Exibe a precisão final do modelo.
- Salva o modelo em `model.p`.

> 🎯 Exemplo de saída:
```
98.75% dos dados foram classificados com sucesso !
```

---

## 🧠🗣️ Etapa 4: Interface com Previsão e Fala (`interface_classifier.py`)

Este é o sistema final interativo que:

1. Inicia a webcam em **tela cheia**.
2. Usa o **MediaPipe** para detectar gestos da mão ao vivo.
3. Normaliza os pontos da mão e faz a predição com o modelo `model.p`.
4. Mostra o gesto reconhecido na tela e desenha a mão.
5. Usa **pyttsx3** para **falar** o gesto detectado com voz em português.

> ✅ Pressione **Q** para encerrar a execução.

### 🗣️ Exemplo de Frases Reconhecidas
- “Oi”
- “Meu”
- “Nome”
- “Matheus”

Estes rótulos são definidos no dicionário:

```python
labels_dict = {0: 'Oi', 1: 'Meu', 2: 'Nome', 3: 'Matheus'}
```

---

## 🧪 Sugestão de Fluxo de Uso

1. Rode `collect_img.py` para capturar imagens por classe.
2. Rode `create_dataset.py` para gerar o arquivo de dataset.
3. Rode `train_classifier.py` para treinar e salvar o modelo.
4. Rode `interface_classifier.py` para usar o sistema ao vivo com previsão e voz.

---

## 💡 Ideias de Expansão

- Adicionar mais classes (gestos).
- Usar CNN em vez de Random Forest para maior precisão.
- Adicionar modo de "frase contínua" com buffer de palavras.
- Exportar frases detectadas para um arquivo `.txt`.

---

## 📚 Créditos

Este projeto utiliza as seguintes tecnologias:

- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)
