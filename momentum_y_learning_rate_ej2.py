# -*- coding: utf-8 -*-
import parser as psr
import perceptron as ppn
import matplotlib.pyplot as plt
from parameters import iniciar

nro_ejercicio, filepath, eta, epochs, capas, train_pct, test_pct, validation_pct, \
    f_activacion, d_pesos, tambatch, momentum, red_desde_archivo, red_hacia_archivo, estop, adaptativo = iniciar()

i = psr.Parser()

datos_train, datos_test, datos_validation = i.parse(nro_ejercicio, train_pct, test_pct, validation_pct)

# Ejemplo de train
DATOS = datos_train
DATOS_TEST = datos_test
DATOS_VALIDACION = datos_validation

N_ENTRADA = len(DATOS[0][0])
RESULTADOS_ESPERADOS = [row[-1] for row in DATOS]

if isinstance(RESULTADOS_ESPERADOS[0], tuple):
    N_SALIDA = len(RESULTADOS_ESPERADOS[0])
else:
    N_SALIDA = 1

rondas = 5
capas = [7]
momentums = [0, 0.05, 0.1, 0.5, 0.9]
etas = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]

for q, eta in enumerate(etas):
    datos_graficos_avg = [[] for i in range(len(momentums))]
    datos_graficos_avg_validacion = [[] for i in range(len(momentums))]

    for k, momentum in enumerate(momentums):
        errores_finales = []
        eficiencias = []
        validaciones = []
        datos_graficos = []

        print "Corriendo " + str(rondas) + " rondas de prueba para momentum " + str(momentum) + " y learning rate " + str(eta)

        for j in range(rondas):

            results = []

            PPN = ppn.PerceptronMulticapa(N_ENTRADA, capas, N_SALIDA, funcion_activacion=f_activacion,
                                        distribucion_pesos=d_pesos, momentum=momentum)
            
            results.append(PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, DATOS_VALIDACION, eta=eta, epochs=epochs,
                                    tamanio_muestra_batch=tambatch, early_stopping_treshold=estop, print_epochs=False))

            resultados = []
            esperados = []
            
            for _ in range(100):
                for fila in DATOS_TEST:
                    prediccion = PPN.predecir_ej2(fila[0])
                    resultados.append(prediccion)

                esperado = [row[-1] for row in DATOS_TEST]
                esperados = esperados + esperado

            eficiencias.append(PPN.medir_performance(esperados, resultados))
            errores_finales.append(results[-1][-1]['funcion_de_costo'])
            validaciones.append(results[-1][-1]['validacion'])

            print "Corrida " + str(j) + ' Error: ' + str(results[-1][-1]['funcion_de_costo']) + \
                ' Eficiencia test: ' + str(PPN.medir_performance(esperados, resultados))

            datos_graficos.append([row['funcion_de_costo'] for row in results[-1]])

            if len(datos_graficos_avg[k]) > 0:
                datos_graficos_avg[k] = [a+b for a, b in zip(datos_graficos_avg[k], datos_graficos[-1])]
            else:
                datos_graficos_avg[k] = (datos_graficos[-1])
            
            if len(datos_graficos_avg_validacion[k]) > 0:
                datos_graficos_avg_validacion[k] = [a+b for a, b in zip(datos_graficos_avg_validacion[k], [row['validacion'] for row in results[-1]])]
            else:
                datos_graficos_avg_validacion[k] = ([row['validacion'] for row in results[-1]])
        
        error_promedio = sum(errores_finales) / rondas
        eficiencia_promedio = sum(eficiencias) / rondas
        error_validaciones = sum(validaciones) / rondas

        print "Error final promediado:" + str(error_promedio)
        print "Error final validaciones: " + str(error_validaciones)
        print "Eficiencia: %.2f %%" % eficiencia_promedio

    # Dibuja el grafico de los promedios de las corridas
    # para cada uno de los momentums
    for i, _ in enumerate(datos_graficos_avg):
        plt.plot([x / float(rondas) for x in datos_graficos_avg[i]], label="Momentum "+str(momentums[i]))

    plt.xlabel('Epocas')
    plt.ylabel('Error/Funcion Costo')
    plt.legend(loc=1)
    plt.savefig('informe/graficos/ej2/eta_x_momentum_promedios_entrenamiento__'+str(q)+'.png')
    plt.clf()

    # Para validaciones
    for i, _ in enumerate(datos_graficos_avg_validacion):
        plt.plot([x / float(rondas) for x in datos_graficos_avg_validacion[i]], label="Momentum "+str(momentums[i]))

    plt.xlabel('Epocas')
    plt.ylabel('Error/Funcion Costo')
    plt.legend(loc=1)
    plt.savefig('informe/graficos/ej2/eta_x_momentum_promedios_validacion__'+str(q)+'.png')
    plt.clf()
