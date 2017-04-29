import numpy as np
import matplotlib.pyplot as plt

class PerceptronMulticapa(object):

    # Constructor de la clase. 
    def __init__(self, nLayersNodes):
        self.pesos = []
        self.n_layers_nodes = nLayersNodes

        # Genera vectores aleatorios para todos los ejes de cada
        # nodo de la red
        for i in range(len(self.n_layers_nodes) - 1):
            self.pesos.append([])
            for _ in range(self.n_layers_nodes[i]):
                self.pesos[i].append(np.random.rand(self.n_layers_nodes[i+1]))

    def activation_function(self, n_layer_value):
        # Funcion de activacion escalon
        return np.where(n_layer_value >= 0.0, 1, 0)

    def layer_input(self, previous_layer_values, n_layer):
        pesos_layer = []

        # Calcula el peso de cada node de la capa numero n_layer
        # a partir de la aplicacion de la funcion de activacion
        # sobre el input de la capa anterior
        for i in range(self.n_layers_nodes[n_layer]):
            pesos_layer.append(self.activation_function(self.node_layer_input(previous_layer_values, i, self.pesos[n_layer-1])))

        return pesos_layer

    def node_layer_input(self, previous_layer_values, node, pesos):
        pesos_node = []

        for i in range(len(pesos)):
            pesos_node.append(pesos[i][node])

        return np.dot(previous_layer_values, pesos_node)

    # "Eta" es el factor de aprendizaje, y "epochs" el numero maximo de epocas de entrenamiento.
    def train(self, input_values, expected_output, eta=0.01, epochs=100):
        current_layer_value = input_values

        for _ in range(epochs):
            
            # Hace el forward primero. 
            for i in range(len(self.n_layers_nodes) - 1):
                current_layer_value = self.layer_input(current_layer_value, i+1)
                print current_layer_value

            # Hace el backwards
            # TODO: implementar backwards

        return self



# Se crea el objeto perceptron.
ppn = PerceptronMulticapa([4, 2, 3, 4])

# Se entrena el perceptron. 
ppn.train(input_values=[0.7, 1.5, -1.3, 4], expected_output=[1, 0, 0, 1], epochs=1, eta=10 ** -1)

# Se grafica la cantidad de clasificaciones incorrectas en cada epoca versus numero de epoca. 
# MEJORAS SOLICITADAS: Mejora 1) Graficar la funcion de costo versus el numero de epoca.
# Mejora 2) Agregar al grafico el error de costo del conjunto de validacion (o sea, debe graficar en el mismo grafico, ambos errores de costo)
#plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epocas')
#plt.ylabel('Clasificaciones erroneas')
#plt.show()

# MEJORAS SOLICITADAS: Agregar imprimir la funcion de costo y el root-mean-square error final del conjunto de testing, 
# para saber la performance final de la red neuronal.