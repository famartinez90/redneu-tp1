import numpy as np

class Normalizador():

	def __init__(self):
		pass

	def normalizar(self, e, media, desvio):
		return (e - media) / desvio

	def desnormalizar(self, e, media, desvio):
		return (e * desvio) + media

	def normalizarArray(self, a):
		
		media = np.mean(a)
		desvio = np.std(a)
		
		normalizados = [self.normalizar(elem, media, desvio) for elem in a]
		
		return media, desvio, normalizados

	def desnormalizarArray(self, a, media, desvio):
		return [self.desnormalizar(elem, media, desvio) for elem in a]
