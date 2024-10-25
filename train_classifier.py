# Importando as bibliotecas necessárias
# Importing required libraries
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carregando os dados do arquivo pickle
# Loading data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convertendo os dados e rótulos para arrays numpy
# Converting data and labels to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividindo os dados em conjuntos de treino e teste (80% treino, 20% teste)
# Splitting data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Criando e treinando o modelo Random Forest
# Creating and training the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Fazendo previsões com o conjunto de teste
# Making predictions with test set
y_predict = model.predict(x_test)

# Calculando a precisão do modelo
# Calculating model accuracy
score = accuracy_score(y_predict, y_test)

# Exibindo a porcentagem de precisão
# Displaying accuracy percentage
print('{}% dos dados foram classificados com sucesso !'.format(score * 100))

# Salvando o modelo treinado em um arquivo pickle
# Saving the trained model to a pickle file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
