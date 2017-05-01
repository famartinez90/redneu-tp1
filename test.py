import perceptron as ppn
import matplotlib.pyplot as plt

# Ejemplo de train
DATOS = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], 
         [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0], 
         [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1], 
         [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], 
         [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]

N_ENTRADA = len(DATOS[0]) - 1
N_SALIDA = len(set([row[-1] for row in DATOS]))
PPN = ppn.PerceptronMulticapa(N_ENTRADA, [5], 2, funcion_activacion="sigmoidea")
PPN.train(DATOS, N_SALIDA, eta=0.5, epochs=50)

DATOS_PREDICCION = [[2.7810836, 2.550537003, 0], 
                    [1.465489372, 2.362125076, 0], 
                    [3.396561688, 4.400293529, 0], 
                    [1.38807019, 1.850220317, 0], 
                    [3.06407232, 3.005305973, 0], 
                    [7.627531214, 2.759262235, 1], 
                    [5.332441248, 2.088626775, 1], 
                    [6.922596716, 1.77106367, 1], 
                    [8.675418651, -0.242068655, 1], 
                    [7.673756466, 3.508563011, 1]]

resultados = []
esperados = []
for _ in range(100):
    for fila in DATOS_PREDICCION:
        prediccion = PPN.predecir(fila)
        resultados.append(prediccion)
    
    esperado = map(lambda xs: xs[-1], DATOS_PREDICCION)
    esperados = esperados + esperado

print "Eficiencia: %.2f %%" % PPN.medir_performance(esperados, resultados)

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

# NOTE: High momentum should always be accompanied by low learning rate, else you will overshoot the global optimum.

# NOTE: Common pitfall. An important point to make about the preprocessing is that any preprocessing statistics (e.g. the data mean)
#  must only be computed on the training data, and then applied to the validation / test data. E.g. computing the mean and subtracting 
# it from every image across the entire dataset and then splitting the data into train/val/test splits would be a mistake. Instead, 
# the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).
