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
momentum = 0
adaptativos = [False, True]
datos_graficos_avg = [[] for i in range(len(adaptativos))]
datos_graficos_avg_validacion = [[] for i in range(len(adaptativos))]

for k, adaptativo in enumerate(adaptativos):
    errores_finales = []
    eficiencias = []
    validaciones = []
    datos_graficos = []

    if adaptativo:
        print "Corriendo " + str(rondas) + " rondas de prueba con parámetros adaptativos"
    else:
        print "Corriendo " + str(rondas) + " rondas de prueba sin parámetros adaptativos"

    for j in range(rondas):

        results = []

        PPN = ppn.PerceptronMulticapa(N_ENTRADA, capas, N_SALIDA, funcion_activacion=f_activacion,
                                      distribucion_pesos=d_pesos, momentum=momentum)
        
        results.append(PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, DATOS_VALIDACION, eta=eta, epochs=epochs,
                                 tamanio_muestra_batch=tambatch, early_stopping_treshold=estop, print_epochs=False, adaptativo=adaptativo))

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

    graficar = True

    if graficar:
        show = []
        labels = []

        for i, _ in enumerate(datos_graficos):
            show.append(range(epochs))
            show.append(datos_graficos[i])
            labels.append("Corrida "+str(i+1))

        
        for i in range(0, len(show), 2):
            plt.plot(show[i], show[i+1], label=labels[i/2])
            
        plt.xlabel('Epocas')
        plt.ylabel('Error/Funcion Costo')
        plt.legend(loc=1)
        
        if adaptativo:
            plt.title("Evolucion del error con parametros adaptativos")
            plt.savefig('graficos/adaptativos_si.png')
        else:
            plt.title("Evolucion del error sin parametros adaptativos")
            plt.savefig('graficos/adaptativos_no.png')
        
        plt.clf()


# Dibuja el grafico de los promedios de las corridas
# para cada uno de los momentums
plt.title("Evolucion del error entrenamiento promedio para Parametros Adaptativos")

for i, _ in enumerate(datos_graficos_avg):
    if adaptativos[i]:
        plt.plot([x / float(rondas) for x in datos_graficos_avg[i]], label="Con Adaptativos")
    else:
        plt.plot([x / float(rondas) for x in datos_graficos_avg[i]], label="Sin Adaptativos")

plt.xlabel('Epocas')
plt.ylabel('Error/Funcion Costo')
plt.legend(loc=1)
plt.savefig('graficos/adaptativos_promedios_entrenamiento.png')
plt.clf()

# Para validaciones
plt.title("Evolucion del error validacion promedio para el Momentum")

for i, _ in enumerate(datos_graficos_avg_validacion):
    if adaptativos[i]:
        plt.plot([x / float(rondas) for x in datos_graficos_avg_validacion[i]], label="Con Adaptativos")
    else:
        plt.plot([x / float(rondas) for x in datos_graficos_avg_validacion[i]], label="Sin Adaptativos")

plt.xlabel('Epocas')
plt.ylabel('Error/Funcion Costo')
plt.legend(loc=1)
plt.savefig('graficos/adaptativos_promedios_validacion.png')
plt.clf()
