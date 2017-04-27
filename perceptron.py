import numpy as np
import matplotlib.pyplot as plt

class PerceptronMulticapa(object):

    # Constructor de la clase. 
    def __init__(self, inputValues, nLayersNodes):
        self.pesos = []
        self.E_0 = inputValues
        self.nLayersNodes = nLayersNodes

        for i in range(len(nLayersNodes) - 1): # [0,1,2]
            self.pesos.append([])
            for j in range(nLayersNodes[i]): # [i=0 => j in [0,1,2,3],[0,1],[0,1,2]]
                self.pesos[i].append(np.random.rand(nLayersNodes[i+1]))
        
        print self.layer_input(1)

    def layer_input(self, nLayer):
        pesosLayer = []

        for i in range(self.nLayersNodes[nLayer]):
            pesosLayer.append(self.node_layer_input(i, self.pesos[nLayer-1]))

        return pesosLayer

    def node_layer_input(self, node, pesos):
        pesosNode = []

        for i in range(len(pesos)):
            pesosNode.append(pesos[i][node])

        return np.dot(self.E_0, pesosNode)

    # "Eta" es el factor de aprendizaje, y "epochs" el numero maximo de epocas de entrenamiento.
    # Los valores de eta y epochs en la primera linea, son los valores por defecto, si no se especifican dichos parametros en la funcion de llamada. 
    def train(self, X, y, eta=0.01, epochs=100):
        self.eta = eta
        self.epochs = epochs
        # Guarda el historial de los errores de clasificacion del conjunto de entremamiento en este vector. 
        # MEJORAS SOLICITADAS: Agregar que tambien se guarde el historial de los errores del conjunto de validacion.
        self.errors_ = []
        # Entrena la red por la cantidad maxima de epocas especificada. 
        # MEJORAS SOLICITADAS: Modificarlo para que pare el entrenamiento cuando la funcion de costo (E[w] de la clase teorica) del conjunto de validacion
        # sea menor a cierto umbral, y no deba necesariamente entrenar la cantidad maxima de epocas.
        for _ in range(self.epochs):
            errors = 0
            # Pasa por la red todos los patrones del conjunto de entrenamiento (o sea, el perceptron entrena 1 epoca). Notar que en este codigo pasa 
            # los patrones de entrenamiento siempre en el mismo orden. 
            # MEJORAS SOLICITADAS: Modificarlo para que el orden de los patrones de entrenamiento sea random en cada epoca 
            # (estrategia simple que mejora la velocidad de aprendizaje de la red).
            for xi, target in zip(X, y):
                # Actualiza los pesos de la red mediante la Regla Delta. 
                # Notar que el factor de aprendizaje eta, es cuanto influye el error actual de la red en la modificacion de los pesos.
                # MEJORAS SOLICITADAS: Modificar la regla Delta del template para que la actualizacion de los pesos sea valida para funciones de 
                # activacion continuas no lineales.
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                # La dimension del patron de entrada se incrementa en 1 en forma ficticia por el bias, y el primer elemento del vector 
                # de entrada es constante=1.
                # Por eso aca, cuando hace la actualizacion del w_[0] (que representa el valor del bias), multiplica update por 1.
                self.w_[0] +=  update * 1
                # MEJORAS SOLICITADAS: Modificar para que calcule y guarde la funcion de costo E[w] de los conjuntos de entrenamiento y de validacion,
                # en lugar de la cantidad de patrones que clasifico mal en esa epoca.
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # Empieza a calcular la salida del perceptron simple.
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Aplica la funcion de activacion (en el caso del template, la funcion escalon unipolar) 
    # MEJORAS SOLICITADAS: Mejora1) Modificar para que implemente otras funciones de activacion, no solo la escalon.
    # Mejora2) Implementar funciones de activacion bipolares.
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Dataset de entrenamiento para el AND logico. 
# MEJORAS SOLICITADAS: Mejora1) Modificarlo para probar con otros datasets de otros problemas.
# Mejora2) Crear tambien conjuntos de valicacion y de testing.
#X = np.array([(0,0), (0,1), (1, 0), (1, 1)])
# Usa salidas deseadas unipolares. 
# MEJORAS SOLICITADAS: Probar con salidas bipolares, y ver si mejora la performance de clasificacion 
# (ojo que si la salida deseada es bipolar, la funcion de activacion de las unidades de salida tambien debe ser bipolar)
#y = np.array([0, 0, 0, 1])

# Dimension de los patrones de entremiento. Por ejemplo, para el AND logico, la dimension de los patrones de entremiento es 2.
#DimTrain = X.shape[1]

# Se crea el objeto perceptron. 
ppn = PerceptronMulticapa([5, 1, 10, 5], [4, 2, 3, 4])

# Se entrena el perceptron. 
# MEJORAS SOLICITADAS: Experimentar con otra cantidad de epocas y factor de aprendizaje.
#ppn.train(X, y, epochs=10, eta=10 ** -1)

# Se grafica la cantidad de clasificaciones incorrectas en cada epoca versus numero de epoca. 
# MEJORAS SOLICITADAS: Mejora 1) Graficar la funcion de costo versus el numero de epoca.
# Mejora 2) Agregar al grafico el error de costo del conjunto de validacion (o sea, debe graficar en el mismo grafico, ambos errores de costo)
#plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epocas')
#plt.ylabel('Clasificaciones erroneas')
#plt.show()

# MEJORAS SOLICITADAS: Agregar imprimir la funcion de costo y el root-mean-square error final del conjunto de testing, 
# para saber la performance final de la red neuronal.