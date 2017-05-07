# -*- coding: utf-8 -*-
import parser as psr
import perceptron as ppn
import matplotlib.pyplot as plt
from script import iniciar

nro_ejercicio, eta, epochs, capas, train_pct, test_pct, validation_pct, \
    f_activacion, d_pesos, tambatch, momentum, red_desde_archivo = iniciar()

i = psr.Parser()

datos_train, datos_validation, datos_test = i.parse(nro_ejercicio, train_pct, test_pct, validation_pct)

# Ejemplo de train
DATOS = datos_train

N_ENTRADA = len(DATOS[0][0])
RESULTADOS_ESPERADOS = [row[-1] for row in DATOS]

if isinstance(RESULTADOS_ESPERADOS[0], tuple):
    N_SALIDA = len(RESULTADOS_ESPERADOS[0])
else:
    N_SALIDA = 1

results = []
for i in [1,2,5,10,30,50]:

    print "Ejecucion con "+str(i)+" capas de 10 neuronas"

    # Las capas seran listas de i veces 10 neuronas. Para la primera corrida, 1 capa de 10, para la 2da 2 capas de 10, tercera 5 capas de 10 etc
    capas = [10] * i
    PPN = ppn.PerceptronMulticapa(N_ENTRADA, capas, N_SALIDA, funcion_activacion=f_activacion, distribucion_pesos=d_pesos, momentum=momentum)
    results.append(PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, eta=eta, epochs=epochs, tamanio_muestra_batch=tambatch))

    DATOS_PREDICCION = [row[0] for row in datos_validation]

    resultados = []
    esperados = []
    for _ in range(100):
        for fila in DATOS_PREDICCION:
            prediccion = PPN.predecir_ej1(fila)
            resultados.append(prediccion)

        esperado = map(lambda row: row[-1], datos_validation)
        esperados = esperados + esperado


    print "Error final:" + results[-1]['error']
    print "Eficiencia: %.2f %%" % PPN.medir_performance(esperados, resultados)