import numpy as np
import matplotlib.pyplot as plt
import math as math

class PerceptronMulticapa(object):

    # Constructor de la clase. 
    def __init__(self, n_entrada, ns_ocultas, n_salida):
        # Inicializamos todos los pesos de los ejes de la
        # red en valores random pequenios, entre 0 y 1, 
        # provenientes de una distribucion normal
        # REF: Algoritmo Backpropagation - Primer paso
        self.pesos_red = list()

        # Calcula pesos de ejes para la primer capa oculta + bias
        pesos_capa_oculta_0 = []
        for _ in range(ns_ocultas[0]):
            pesos_capa_oculta_0.append({'pesos': np.random.rand(n_entrada + 1)})
        
        self.pesos_red.append(pesos_capa_oculta_0)
        
        # Calcula pesos de ejes para siguientes capas ocultas + bias
        pesos_capas_ocultas = []
        for i in range(len(ns_ocultas) - 1):

            for _ in range(ns_ocultas[i+1]):
                pesos_capas_ocultas.append({'pesos': np.random.rand(ns_ocultas[i] + 1)})
            
            self.pesos_red.append(pesos_capas_ocultas)

        # Calcula pesos de ejes para las neuronas de salida
        pesos_capa_salida = []
        for _ in range(n_salida):
            pesos_capa_salida.append({'pesos': np.random.rand(ns_ocultas[len(ns_ocultas) - 1] + 1)})
        
        self.pesos_red.append(pesos_capa_salida)

    def funcion_de_suma(self, pesos, entrada):
        # Quito el bias para calcular
        suma = pesos[-1]
        
        for i in range(len(pesos) - 1):
            suma += pesos[i] * entrada[i]
        
        return suma

    def funcion_de_activacion(self, suma):
        # Usa la funcion sigmoidea
        # g = 1 / 1 + e^-x
        return 1.0 / (1.0 + math.exp(-suma))

    def derivada_funcion_de_activacion(self, suma):
        # Esta es la derivada de la sigmoidea
        # g' = x * (1 - x)
        return suma * (1.0 - suma)

    def propagacion_forward(self, valores_de_entrada):
        # REF: Algoritmo Backpropagation - 6.15
        salida = valores_de_entrada
        
        # Para cada capa (exceptuando la de entrada) calcula el valor de salida
        # de cada una de sus neuronas aplicando la funcion de activacion sobre
        # la suma de los ejes de la capa a anterior que llegan a esa neurona
        # REF: Algoritmo Backpropagation - 6.16
        for capa in self.pesos_red:
            nueva_salida = []
            
            for neurona in capa:
                suma = self.funcion_de_suma(neurona['pesos'], salida)
                neurona['salida'] = self.funcion_de_activacion(suma)
                nueva_salida.append(neurona['salida'])
            
            # Al pisar el valor de salida con el de la capa que se acaba de
            # calcular se propaga hacia adelante para poder calcular la proxima capa
            salida = nueva_salida
        
        return salida

    def propagacion_backward(self, salida_esperada):
        for i in reversed(range(len(self.pesos_red))):
            capa = self.pesos_red[i]
            errores = list()
            
            if i != len(self.pesos_red)-1:
                # Si no es la capa de salida, entonces propagamos el error
                # hacia atras sobre las neuronas de mi capa actual, donde el error
                # de mi neurona es la suma del producto de cada uno de mis ejes por
                # el delta de la neurona a la que llega ese eje
                # REF: Algoritmo Backpropagation - 6.18
                for j in range(len(capa)):
                    error = 0.0
                    neuronas_capa_siguiente = self.pesos_red[i + 1]
                    
                    for neurona in neuronas_capa_siguiente:
                        error += (neurona['pesos'][j] * neurona['delta'])
                    
                    errores.append(error)
            
            else:
                # Si es la capa de salida entonces calculamos el error a partir
                # de la diferencia entre el valor de la salida de la neurona
                # y el valor esperado de salida del dataset
                # REF: Algoritmo Backpropagation - 6.17
                for j, _ in enumerate(capa):
                    neurona = capa[j]
                    errores.append(salida_esperada[j] - neurona['salida'])
            
            # Calcula el delta para cada neurona de la capa actual
            # haciendo el producto del error por la derivada de la funcion
            # de activacion
            # REF: Algoritmo Backpropagation - 6.17/6.18
            for j, _ in enumerate(capa):
                neurona = capa[j]
                neurona['delta'] = errores[j] * self.derivada_funcion_de_activacion(neurona['salida'])

    def actualizar_pesos(self, fila_dataset, eta):
        # Actualiza los pesos para toda la red
        for i, _ in enumerate(self.pesos_red):
            entrada = fila_dataset[:-1]
            
            if i != 0:
                # Voy pisando el valor de 'entrada' con los valores de 
                # salida de la capa actual en la que estoy para luego
                # calcular los deltas correctamente
                salida_neuronas_capa_anterior = []

                for neurona in self.pesos_red[i - 1]:
                    salida_neuronas_capa_anterior.append(neurona['salida'])

                entrada = salida_neuronas_capa_anterior
            
            # Para cada neurona de la capa actual actualizo los pesos
            # utilizando la formula delta
            for neurona in self.pesos_red[i]:
                
                for j, _ in enumerate(entrada):
                    # Aqui aplico la formula delta
                    # REF: Algoritmo Backpropagation - 6.19
                    neurona['pesos'][j] += eta * neurona['delta'] * entrada[j]
                
                # Aqui actualizo el bias
                neurona['pesos'][-1] += eta * neurona['delta']

    # "Eta" es el factor de aprendizaje, y "epochs" el numero maximo de epocas de entrenamiento.
    def train(self, dataset, n_salida, eta=0.01, epochs=50000):
        # Para cada epoca, paso por cada una de las filas de entrada del dataset
        # y actualizo los pesos de la red con la regla delta, como la actualizacion
        # se produce cada vez que paso por una fila nueva, no es batch. Para hacerlo
        # batch deberia acumular el error de mas de una fila antes de correr 
        # actualizar_pesos
        for epoch in range(epochs):
            error_acumulado = 0
            
            for fila in dataset:
                salida = self.propagacion_forward(fila)
                esperado = [0 for i in range(n_salida)]
                esperado[fila[-1]] = 1
                error_acumulado += sum([(esperado[i] - salida[i]) ** 2 for i, _ in enumerate(esperado)])
                self.propagacion_backward(esperado)
                self.actualizar_pesos(fila, eta)
            
            print('>epoch=%d, eta=%.3f, error_acumulado=%.3f' % (epoch, eta, error_acumulado))




class InputParser():
	def parsear():
		pass

# Se crea el objeto perceptron.
ppn = PerceptronMulticapa([4, 2, 3, 4])

# Ejemplo de train
DATOS = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1], [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]
N_ENTRADA = len(DATOS[0]) - 1
N_SALIDA = len(set([row[-1] for row in DATOS]))
PPN = PerceptronMulticapa(N_ENTRADA, [2], 2)
PPN.train(DATOS, N_SALIDA)


# print PPN.propagacion_forward([17.673756466, 13.508563011, 1])

# Se grafica la cantidad de clasificaciones incorrectas en cada epoca versus numero de epoca. 
# MEJORAS SOLICITADAS: Mejora 1) Graficar la funcion de costo versus el numero de epoca.
# Mejora 2) Agregar al grafico el error de costo del conjunto de validacion (o sea, debe graficar en el mismo grafico, ambos errores de costo)
#plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epocas')
#plt.ylabel('Clasificaciones erroneas')
#plt.show()

# MEJORAS SOLICITADAS: Agregar imprimir la funcion de costo y el root-mean-square error final del conjunto de testing, 
# para saber la performance final de la red neuronal.