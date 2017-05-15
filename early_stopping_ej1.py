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
capas = [10, 10]
early_stoppings = [0, 0.2, 0.15, 0.1, 0.05, 0.01]
eficiencia_testing = []

for k, estop in enumerate(early_stoppings):
    errores_finales = []
    eficiencias = []
    validaciones = []
    datos_graficos_entrenamiento = []
    datos_graficos_validacion = []

    print "Corriendo " + str(rondas) + " rondas de prueba para early stopping treshold " + str(estop)

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
                prediccion = PPN.predecir_ej1(fila[0])
                resultados.append(prediccion)

            esperado = [row[-1] for row in DATOS_TEST]
            esperados = esperados + esperado

        eficiencias.append(PPN.medir_performance(esperados, resultados))
        errores_finales.append(results[-1][-1]['funcion_de_costo'])
        validaciones.append(results[-1][-1]['validacion'])


        print "Corrida " + str(j) + ' Error: ' + str(results[-1][-1]['funcion_de_costo']) + \
            ' Eficiencia test: ' + str(PPN.medir_performance(esperados, resultados))

        datos_graficos_entrenamiento.append([row['funcion_de_costo'] for row in results[-1]])
        datos_graficos_validacion.append([row['validacion'] for row in results[-1]])

    error_promedio = sum(errores_finales) / rondas
    eficiencia_promedio = sum(eficiencias) / rondas
    error_validaciones = sum(validaciones) / rondas

    eficiencia_testing.append(eficiencia_promedio)
    
    print "Error final promediado:" + str(error_promedio)
    print "Error final validaciones: " + str(error_validaciones)
    print "Eficiencia: %.2f %%" % eficiencia_promedio

    graficar = True

    if graficar:
        show = []
        labels = []

        for i, _ in enumerate(datos_graficos_entrenamiento):
            show.append(range(epochs))
            show.append(datos_graficos_entrenamiento[i] + [datos_graficos_entrenamiento[i][-1]] * (epochs - len(datos_graficos_entrenamiento[i])))
            labels.append("Corrida "+str(i+1))

        for i in range(0, len(show), 2):
            plt.plot(show[i], show[i+1], label=labels[i/2])
            
        plt.xlabel('Epocas')
        plt.ylabel('Error/Funcion Costo')
        plt.legend(loc=1)
        plt.savefig('informe/graficos/early_stopping_entrenamiento_'+str(estop)+'.png')
        plt.clf()

    if graficar:
        show = []
        labels = []

        for i, _ in enumerate(datos_graficos_validacion):
            show.append(range(epochs))
            show.append(datos_graficos_validacion[i] + [datos_graficos_validacion[i][-1]] * (epochs - len(datos_graficos_validacion[i])))
            labels.append("Corrida "+str(i+1))

        for i in range(0, len(show), 2):
            plt.plot(show[i], show[i+1], label=labels[i/2])
            
        plt.xlabel('Epocas')
        plt.ylabel('Error/Funcion Costo')
        plt.legend(loc=1)
        plt.savefig('informe/graficos/early_stopping_validacion_'+str(estop)+'.png')
        plt.clf()


plt.bar(range(len(early_stoppings)), eficiencia_testing, width=0.5)
plt.xticks(range(len(early_stoppings)), early_stoppings)
plt.yticks(range(0, 110, 10))
plt.xlabel('Early Stopping Treshold')
plt.ylabel('Error/Funcion Costo')
plt.savefig('informe/graficos/early_stopping_testing.png')
plt.clf()
