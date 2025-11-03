import numpy as np

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.pesos = None
        self.sesgo = None
        self.tasa_aprendizaje = learning_rate
        self.epochs = epochs

    # Función de activacion Escalón Unitario
    def funcion_de_activacion(self, z):
        return np.heaviside(z, 0) # haviside(z) heaviside -> activación

    def fit(self, X, y):
        n_caracteristicas = X.shape[1]
        
        # Inicializando pesos y sesgo
        self.pesos = np.zeros((n_caracteristicas))
        self.sesgo = 0
        
        # Iteraciones hasta que se alcancen el número de épocas
        for epoch in range(self.epochs):
            
            # Recorrido por todo el conjunto de entrenamiento
            for i in range(len(X)):
                z = np.dot(X, self.pesos) + self.sesgo # Realiza el producto punto y suma el sesgo
                y_pred = self.funcion_de_activacion(z) # Aplicando la función de activación
                
                # Actualizando pesos y sesgo
                self.pesos = self.pesos + self.tasa_aprendizaje * (y[i] - y_pred[i]) * X[i]
                self.sesgo = self.sesgo + self.tasa_aprendizaje * (y[i] - y_pred[i])
                
        return self.pesos, self.sesgo

    def prediccion(self, X):
        z = np.dot(X, self.pesos) + self.sesgo
        return self.funcion_de_activacion(z)        
