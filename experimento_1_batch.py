# -*- coding: utf-8 -*-
import parser as psr
import perceptron as ppn
import matplotlib.pyplot as plt
from parameters import iniciar

nro_ejercicio, eta, epochs, capas, train_pct, test_pct, validation_pct, \
    f_activacion, d_pesos, tambatch, momentum, red_desde_archivo, estop, adaptativo = iniciar()

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

debug = False
csv = True
rondas = 5
capas = [5]

if csv:
    print "Ejecución" + ', ' + "Error Final" + ', ' + "Error Validación" + ', ' + "Eficiencia Testing"

# tamanio_muestra_batch = 1 ---> online learning
# 1 < tamanio_muestra_batch <= len(dataset) / 2 ---> mini batch
# tamanio_muestra_batch = len(dataset) ---> batch

for i in [1, 5, 10, 100, 'mitad', 'batch']:

    if debug:
        print "Se ejecutarán " + str(rondas) + " rondas con" + str(len(i)) +"capas de 10 neuronas"

    for inicializacion_pesos in ['uniforme', 'normal']:

        errores_finales = []
        eficiencias = []
        validaciones = []

        for j in range(rondas):

            results = []

            if i == 'mitad':
                tambatch = len(DATOS) / 2
            elif i == 'batch':
                tambatch = len(DATOS)
            else:
                tambatch = i

            PPN = ppn.PerceptronMulticapa(N_ENTRADA, capas, N_SALIDA, funcion_activacion=f_activacion,
                                          distribucion_pesos=inicializacion_pesos, momentum=momentum)

            results.append(PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, datos_validation, eta=eta, epochs=epochs,
                                     tamanio_muestra_batch=tambatch, early_stopping_treshold=estop, print_epochs=False))

            DATOS_PREDICCION = [row[0] for row in datos_test]

            resultados = []
            esperados = []
            for _ in range(100):
                for fila in DATOS_PREDICCION:

                    if nro_ejercicio == '1':
                        prediccion = PPN.predecir_ej1(fila)
                    else:
                        prediccion = PPN.predecir_ej2(fila)

                    resultados.append(prediccion)

                esperado = map(lambda row: row[-1], datos_test)
                esperados = esperados + esperado


            performance = PPN.medir_performance(esperados, resultados)
            eficiencias.append(performance)
            errores_finales.append(results[-1][-1]['funcion_de_costo'])
            validaciones.append(results[-1][-1]['validacion'])

            if debug:
                print "Corrida " + str(j) + ' Error: ' + str(results[-1][-1]['funcion_de_costo']) + \
                      ' Error Validacion: ' + str(results[-1][-1]['validacion']) + \
                      ' Eficiencia: ' + str(performance)

        error_promedio = sum(errores_finales) / rondas
        eficiencia_promedio = sum(eficiencias) / rondas
        error_validaciones = sum(validaciones) / rondas

        if debug:
            print "Error final promediado:" + str(error_promedio)
            print "Error final validaciones: " + str(error_validaciones)
            print "Eficiencia: %.2f %%" % eficiencia_promedio

        if csv:
            print "Batch: "+ str(i) + ' Init:' + str(inicializacion_pesos) + ', ' + str(error_promedio) + ', ' + str(error_validaciones) + ', ' + str(eficiencia_promedio)