# -*- coding: utf-8 -*-
import parser as psr
import perceptron as ppn
import matplotlib.pyplot as plt
from parameters import iniciar

nro_ejercicio, eta, epochs, capas, train_pct, test_pct, validation_pct, \
    f_activacion, d_pesos, tambatch, momentum, red_desde_archivo, estop = iniciar()

i = psr.Parser()

datos_train, datos_test, datos_validation = i.parse(nro_ejercicio, train_pct, test_pct, validation_pct)

# Ejemplo de train
DATOS = datos_train

N_ENTRADA = len(DATOS[0][0])
RESULTADOS_ESPERADOS = [row[-1] for row in DATOS]

if isinstance(RESULTADOS_ESPERADOS[0], tuple):
    N_SALIDA = len(RESULTADOS_ESPERADOS[0])
else:
    N_SALIDA = 1

rondas = 10

for i in [[10, 10], [15, 15],[20, 20]]:
    errores_finales = []
    eficiencias = []

    print "Se ejecutarán " + str(rondas) + " rondas con 2 capas de "+ str(i[0]) +" neuronas"

    for j in range(rondas):

        results = []

        PPN = ppn.PerceptronMulticapa(N_ENTRADA, i, N_SALIDA, funcion_activacion=f_activacion,
                                      distribucion_pesos=d_pesos, momentum=momentum)
        results.append(PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, datos_validation, eta=eta, epochs=500,
                                 tamanio_muestra_batch=tambatch, early_stopping_treshold=estop, print_epochs=False))

        DATOS_PREDICCION = [row[0] for row in datos_test]

        resultados = []
        esperados = []
        for _ in range(100):
            for fila in DATOS_PREDICCION:
                prediccion = PPN.predecir_ej1(fila)
                resultados.append(prediccion)

            esperado = map(lambda row: row[-1], datos_test)
            esperados = esperados + esperado

        eficiencias.append(PPN.medir_performance(esperados, resultados))
        errores_finales.append(results[-1][-1]['funcion_de_costo'])

        print "Corrida " + str(j) + ' Error: ' + str(results[-1][-1]['funcion_de_costo']) + \
              ' Eficiencia: ' + str(PPN.medir_performance(esperados, resultados))

    error_promedio = sum(errores_finales) / rondas
    eficiencia_promedio = sum(eficiencias) / rondas

    print "Error final promediado:" + str(error_promedio)
    print "Eficiencia: %.2f %%" % eficiencia_promedio