# -*- coding: utf-8 -*-
import csv
import random
import ast

class Parser(object):

	def is_numberic(self, string):
	    try:
	        float(string)
	        return True
	    except ValueError:
	        return False

	def parse(self, nro_ejercicio, train_pct, test_pct, validation_pct):

		if nro_ejercicio == '1':
			filepath = 'tp1_ej1_training.csv'
		else:
			filepath = 'tp1_ej2_training.csv'

		reader = csv.reader(open(filepath, 'r'))
		cant_datos = len(open(filepath).readlines())

		# Estimo cantidad de registros que ira a cada set de datos
		cant_datos_training = int(cant_datos * float(train_pct) / 100)
		cant_datos_test = int(cant_datos * float(test_pct) / 100)
		cant_datos_validation = int(cant_datos * float(validation_pct) / 100)

		datos = []
		datos_train = []
		datos_test = []
		datos_validation = []

		# Paso CSV a lista de rows para manejar mas facil
		for row in reader:
			values = []
			for value in row:
				if self.is_numberic(value):
					value = float(value)
				elif value == 'B':
					value = 1
				elif value == 'M':
					value = -1
				values.append(value)

			datos.append(values)

		# Mezclo datos para que la seleccion y division sean azarosas
		random.shuffle(datos)

		row_number = 0
		for row in datos:
		# Divido los datos del input en porciones respectivas de training, test y validation
			index_training = (0, cant_datos_training)
			index_test = (index_training[1], index_training[1] + cant_datos_test)
			index_validation = (index_test[1], index_test[1] + cant_datos_validation)

			# Separo la salida de cada row. Ahora row con salida sera una tupla
			# Donde el primer elemento es un array de datos y el segundo es el resultado
			# Para el ej1, el resultado es un B = 1 o M = -1
			# Para el ej2, el resultado es una tupla de valores, uno para cada carga
			row_con_salida = self.separar_salida(nro_ejercicio, row)

			if row_number in range(*index_training):
				datos_train.append(row_con_salida)
			elif row_number in range(*index_test):
				datos_test.append(row_con_salida)
			elif row_number in range(*index_validation):
				datos_validation.append(row_con_salida)

			row_number += 1

		return datos_train, datos_validation, datos_test

	def separar_salida(self, nro_ejercicio, row):
		# Se parsea distinto si es el ejercicio 1 o el 2
		# En el ejercicio 1 los resultados son B o M (maligno/benigno)
		# y la salida es 1 columna
		if nro_ejercicio == '1':
			salida = row[0]
			del row[0]
			return (row, salida)

		# En el ejercicio 2 los resultados son B o M (maligno/benigno)
		# y la salida es 1 columna
		elif nro_ejercicio == '2':
			salida = (row[-2], row[-1])
			del row[-2]
			del row[-1]
			return (row, salida)



		# NOTE: Common pitfall. An important point to make about the preprocessing is that any preprocessing statistics (e.g. the data mean)
		#  must only be computed on the training data, and then applied to the validation / test data. E.g. computing the mean and subtracting
		# it from every image across the entire dataset and then splitting the data into train/val/test splits would be a mistake. Instead,
		# the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).

        # http://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
		# # Standardize the data attributes for the Iris dataset.
		# from sklearn.datasets import load_iris
		# from sklearn import preprocessing
		# # load the Iris dataset
		# iris = load_iris()
		# print(iris.data.shape)
		# # separate the data and target attributes
		# X = iris.data
		# y = iris.target
		# # standardize the data attributes
		# standardized_X = preprocessing.scale(X)