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

class Normalizador_2(object):
	""" 
		a = Normalizador_2([2,3,4,...])

		a.original 		-> lista original transformada a array
		a.normalizado 	-> array normalizado
		a.desnormalizado -> array desnormalizado
	"""
	
	def __init__(self, lista):
		
		self.original = np.array(lista)
		self.normalizado = self.normalizar() 
		self.desnormalizado = self.desnormalizar()

	def normalizar(self):
		
		self.media, self.desvio = np.mean(self.original), np.std(self.original)
		return [(( x - self.media ) / self.desvio ) for x in self.original ]

	def desnormalizar(self):
		return [ ((x * self.desvio) + self.media) for x in self.normalizado ]

	

		



#originales = [[-1000, -75, 25, 70, 1000], [30, 75, 0, -20, 1000], [-500, -300, 100, 300, 500]]


#a = [-1000, -75, 25, 70, 1000]


#media, desvio, normalizados = normalizarArray(a)
#print media, desvio, normalizados
#print desnormalizarArray(normalizados, media, desvio)


#medias = map(lambda row: np.mean(row) , originales)
#desvios = map(lambda row: np.std(row) , originales)

#print medias
#print desvios

#for i, normalizado in enumerate(normalizados):
#	print [desnormalizar(elem, medias[i], desvios[i]) for elem in normalizado]


