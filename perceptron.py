# -*- coding: utf-8 -*-
import math as math
import numpy as np
from copy import copy, deepcopy

class PerceptronMulticapa(object):

    # Constructor de la clase. 
    def __init__(self, n_entrada, ns_ocultas, n_salida, funcion_activacion="", distribucion_pesos="",
                 momentum=0, basic_init=False, basic_init_pesos={}):
        # Inicializamos todos los pesos de los ejes de la
        # red en valores random pequeños, entre 0 y 1, 
        # provenientes de una distribución normal
        # REF: Algoritmo Backpropagation - Primer paso
        self.pesos_red = list()
        self.activacion_elegida = funcion_activacion
        self.bias = 1.0
        self.distribucion = distribucion_pesos
        self.momentum = momentum

        if basic_init:
            self.pesos_red = basic_init_pesos
            return

        # Calcula pesos de ejes para la primer capa oculta + bias (último peso)
        pesos_capa_oculta_0 = []
        for _ in range(ns_ocultas[0]):
            pesos_capa_oculta_0.append({'pesos': self.generate_pesos_random(n_entrada + 1), 'delta_w_anterior': 0})
        
        self.pesos_red.append(pesos_capa_oculta_0)

        # Calcula pesos de ejes para siguientes capas ocultas + bias (último peso)
        for i in range(len(ns_ocultas) - 1):
            pesos_capa_oculta_i = []

            for _ in range(ns_ocultas[i+1]):
                pesos_capa_oculta_i.append({'pesos': self.generate_pesos_random(ns_ocultas[i] + 1), 'delta_w_anterior': 0})
            
            self.pesos_red.append(pesos_capa_oculta_i)
        
        # Calcula pesos de ejes para las neuronas de salida + bias (último peso)
        pesos_capa_salida = []
        for _ in range(n_salida):
            pesos_capa_salida.append({'pesos': self.generate_pesos_random(ns_ocultas[len(ns_ocultas) - 1] + 1), 'delta_w_anterior': 0})
        
        self.pesos_red.append(pesos_capa_salida)

    def generate_pesos_random(self, entradas_neurona):
		return {
			# Genera pesos con una distribucion uniforme [0, 1)
			'uniforme': np.random.rand(entradas_neurona).tolist(),
			# Genera pesos con una distribucion normal con media 0 y 
			# varianza entradas_neurona^-1/2, basado en Efficient BackProp
			# de Yann LeCun, fórmula 15, inicialización de pesos eficiente
			# para funciones de activación sigmoideas
			# 'normal': np.random.normal(0, math.sqrt(1.0 / entradas_neurona), entradas_neurona).tolist(),
			'normal': np.random.uniform(-0.1, 0.1, entradas_neurona).tolist(),
		}.get(self.distribucion, np.random.rand(entradas_neurona))

    def funcion_de_suma(self, pesos, entrada):
        suma = pesos[-1] * self.bias

        for i in range(len(pesos) - 1):
            suma += pesos[i] * entrada[i]

        return suma

    def funcion_de_activacion(self, suma):
        return {
            'logistica': self.funcion_logistica,
            'tangente': self.funcion_tangente_hiperbolica,
            'tangente_optimizada': self.funcion_tangente_hiperbolica_optimizada
        }[self.activacion_elegida](suma)

    def funcion_logistica(self, x):
        # Usa la funcion logistica
        # g = 1 / 1 + e^-x
        return 1.0 / (1.0 + math.exp(-x))

    def funcion_tangente_hiperbolica(self, x):
		# Usa la funcion tangente hiperbolica
		# g = (2 / 1 + e^-2x) - 1
		return np.tanh(x)

	def funcion_tangente_hiperbolica_optimizada(self, x):
		# Usa la funcion tangente hiperbolica optimizada
		# g = alfa * ((2 / 1 + e^-(2*beta*x)) - 1)
		alfa = 1.7159
		beta = 2.0 / 3.0
		return alfa * np.tanh(beta * x)

    def derivada_funcion_de_activacion(self, suma):
        return {
            'logistica': self.funcion_logistica_derivada,
            'tangente': self.funcion_tangente_hiperbolica_derivada,
            'tangente_optimizada': self.funcion_tangente_hiperbolica_derivada_optimizada
        }[self.activacion_elegida](suma)

    def funcion_logistica_derivada(self, fx):
		# Esta es la derivada de la logistica
		# g' = g(x) * (1 - g(x))
		return fx * (1.0 - fx)

	def funcion_tangente_hiperbolica_derivada(self, fx):
		# Esta es la derivada de la tangente hiperbolica
		# g = 1 - g(x)^2
		return 1.0 - (fx ** 2)

	def funcion_tangente_hiperbolica_derivada_optimizada(self, fx):
		# Esta es la derivada de la tangente hiperbolica optimizada
		# g = alfa * beta * (1 - g(x)^2)
		alfa = 1.7159
		beta = 2.0 / 3.0
		return alfa * beta * (1.0 - (fx ** 2))

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
            
            if i != len(self.pesos_red) - 1:
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
                for j, neurona in enumerate(capa):
                    if isinstance(salida_esperada[0], tuple): 
                        errores.append(salida_esperada[0][j] - neurona['salida'])
                    else:
                        errores.append(salida_esperada[j] - neurona['salida'])
            
            # Calcula el delta para cada neurona de la capa actual
            # haciendo el producto del error por la derivada de la funcion
            # de activacion
            # REF: Algoritmo Backpropagation - 6.17/6.18
            for j, neurona in enumerate(capa):
                neurona['delta'] = errores[j] * self.derivada_funcion_de_activacion(neurona['salida'])

    def calcular_gradientes(self, fila_dataset, eta, actualizar=True):
        # Actualiza los pesos para toda la red
        # si la variable actualizar es true
        gradientes = []

        for i, _ in enumerate(self.pesos_red):
            entrada = fila_dataset
            
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
                    # Aqui aplico la formula delta para
                    # actualizar el peso
                    # REF: Algoritmo Backpropagation - 6.19
                    delta_w = eta * neurona['delta'] * entrada[j]

                    if self.momentum != 0:
                        delta_w += self.momentum * neurona['delta_w_anterior']

                    if actualizar:
                        neurona['pesos'][j] += delta_w
                    else:
                        gradientes.append(delta_w)
                    
                    neurona['delta_w_anterior'] = delta_w
                
                # Aqui actualizo los pesos del bias
                if actualizar:
                    neurona['pesos'][-1] += eta * neurona['delta']
                else:
                    gradientes.append(eta * neurona['delta'])

        if actualizar is False:
            return gradientes

    # "Eta" es el factor de aprendizaje, y "epochs" el numero maximo de epocas de entrenamiento.
    def train(self, dataset, salida_esperada, validacion, eta=0.5, epochs=100,
              tamanio_muestra_batch=1, adaptativo=False, early_stopping_treshold=0, print_epochs=True):
        # Para cada epoca, paso por cada una de las filas de entrada del dataset
        # y actualizo los pesos de la red con la regla delta
        # tamanio_muestra_batch = 1 ---> online learning
        # 1 < tamanio_muestra_batch <= len(dataset) / 2 ---> mini batch
        # tamanio_muestra_batch = len(dataset) ---> batch

        results = []
        error_anterior = 0.0
        imagen_de_la_red = []
        fluctuacion_del_error = 0

        for epoch in range(epochs):
            funcion_de_costo = 0
            muestra_numero = 0
            gradientes = []

            if adaptativo:
                imagen_de_la_red = deepcopy(self.pesos_red)

            for i, _ in enumerate(self.pesos_red):    
                for neurona in self.pesos_red[i]:
                    for _ in enumerate(neurona['pesos']):
                        gradientes.append([])
            
            for k, fila in enumerate(dataset):
                salida = self.propagacion_forward(fila)
                error_cuadratico = []

                if isinstance(salida_esperada[k], tuple):
                    for i, esperada in enumerate(salida_esperada[k]):
                        error_cuadratico.append((esperada - salida[i]) ** 2)
                else:
                    error_cuadratico.append((salida_esperada[k] - salida[0]) ** 2)

                funcion_de_costo += sum(error_cuadratico)
                self.propagacion_backward([salida_esperada[k]])
                
                if tamanio_muestra_batch == 1:
                    self.calcular_gradientes(fila, eta)
                else:
                    gradientes = [a+[b] for a, b in zip(gradientes, self.calcular_gradientes(fila, eta, False))]
                    muestra_numero += 1

                    if muestra_numero == tamanio_muestra_batch:
                        muestra_numero = 0
                        gradientes_promedios = [np.average(a) for a in gradientes]
                        self.actualizar_pesos_batch(gradientes_promedios)
                

            funcion_de_costo = (funcion_de_costo / 2.0) / len(dataset)

            error_validacion = self.calcular_error_validacion(validacion)

            # Si el learning rate es adaptativo, aca
            # es donde lo va variando segun como evoluciona
            # el error durante las epocas
            if adaptativo:
                if epoch == 1:
                    error_anterior = funcion_de_costo
                else:
                    if funcion_de_costo != 0.0:
                        if funcion_de_costo > error_anterior:
                            fluctuacion_del_error += 1
                        elif funcion_de_costo < error_anterior:
                            fluctuacion_del_error -= 1
                            
                        if fluctuacion_del_error > 1:
                            fluctuacion_del_error = 0
                            self.pesos_red = imagen_de_la_red
                            eta *= 0.5
                        elif fluctuacion_del_error < -1:
                            fluctuacion_del_error = 0
                            self.pesos_red = imagen_de_la_red
                            eta *= 1.1

                        error_anterior = funcion_de_costo

            results.append({'epoca': epoch, 'eta': eta, 'funcion_de_costo': funcion_de_costo})

            if print_epochs:
                print 'epoca: %d, eta: %.3f, error: %.5f, validacion: %.5f' % (epoch, eta, funcion_de_costo, error_validacion)

            if early_stopping_treshold > 0.0:
                if error_validacion >= early_stopping_treshold and funcion_de_costo < 10.0:
                    retrain = False
                    break

        return results

    def actualizar_pesos_batch(self, gradientes_promedios):
        k = 0

        for i, _ in enumerate(self.pesos_red):
            
            for neurona in self.pesos_red[i]:
                
                for j, _ in enumerate(neurona['pesos']):
                    neurona['pesos'][j] += gradientes_promedios[k]
                    k += 1

    # Realiza una prediccion sobre una entrada
    # a partir de una red entrenada
    def predecir_test(self, fila):
        salida = self.propagacion_forward(fila)
        
        if salida[0] > 0.5:
            return 1
        else:
            return 0

    # Realiza una prediccion sobre una entrada
    # a partir de una red entrenada
    def predecir_ej1(self, fila):
        salida = self.propagacion_forward(fila)

        if salida[0] > 0.0:
            return 1
        else:
            return -1

    def calcular_error_validacion(self, datos_validacion):
		entrada = [row[0] for row in datos_validacion]
		esperados = [row[-1] for row in datos_validacion]
		resultados = []
		funcion_de_costo = 0;

		for k, fila in enumerate(entrada):
			error_cuadratico = []
			salida = self.propagacion_forward(fila)

			if isinstance(esperados[k], tuple):
				for i, esperada in enumerate(esperados[k]):
					error_cuadratico.append((esperada - salida[i]) ** 2)
			else:
				error_cuadratico.append((esperados[k] - salida[0]) ** 2)

			funcion_de_costo += sum(error_cuadratico)

		return (funcion_de_costo / 2.0) / len(entrada)

		# for fila in entrada:
		#     prediccion = self.predecir_ej1(fila)
		#     resultados.append(prediccion)

		# return self.medir_performance(esperados, resultados)

    # Permite medir la performance de la red para
    # realizar predicciones a partir de los resultados
    # esperados y la salida de la prediccion
    def medir_performance(self, esperado, salida):
        acertados = 0
        
        for i, _ in enumerate(salida):
            if int(round(esperado[i])) == salida[i]:
                acertados += 1
    	
        return acertados / float(len(salida)) * 100.0
