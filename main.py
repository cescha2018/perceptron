from perceptron import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np


iris = load_iris() #Carga del conjunto de datos Iris

X = iris.data[:, (0, 1)] # petal length, petal width
y = (iris.target == 0).astype(np.int64)

# Dividimos el conjunto de datos presente en nuestro Dataset para entrenar y realizar pruebas
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.5, random_state=42)

perceptron = Perceptron(0.001, 100)
perceptron.fit(X_entrenamiento, y_entrenamiento)
prediccion = perceptron.predict(X_prueba)

# Medición y Puntaje sobre la predicción de modelo
puntaje = accuracy_score(prediccion, y_prueba)
print(f"Puntaje del modelo Perceptron: {puntaje*100:.0f}%")

# Reporte de la precisión del modelo
reporte = classification_report(prediccion, y_prueba, digits=2)
print(reporte)