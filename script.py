# -*- coding: utf-8 -*-
import parameters as params
import parser as psr
import perceptron as ppn
import encoder as encoder
import matplotlib.pyplot as plt

######### INICIO SCRIPT ##############

# Ejemplo de ejecucion:

nro_ejercicio, eta, epochs, capas, train_pct, test_pct, validation_pct, \
    f_activacion, d_pesos, tambatch, momentum, red_desde_archivo, estop = params.iniciar()

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

results = []
for i in range(1):
    if red_desde_archivo is not None:
        PPN = encoder.from_json(red_desde_archivo)
    else:
        PPN = ppn.PerceptronMulticapa(N_ENTRADA, capas, N_SALIDA, funcion_activacion=f_activacion, distribucion_pesos=d_pesos, momentum=momentum)
        results.append(PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, datos_validation, eta=eta, epochs=epochs, tamanio_muestra_batch=tambatch, early_stopping_treshold=estop))

    DATOS_PREDICCION = [row[0] for row in datos_test]

    resultados = []
    esperados = []
    for _ in range(10):
        for fila in DATOS_PREDICCION:
            prediccion = PPN.predecir_ej1(fila)
            resultados.append(prediccion)

        esperado = [row[-1] for row in datos_test]
        esperados = esperados + esperado

    print "Eficiencia: %.2f %%" % PPN.medir_performance(esperados, resultados)

graficar = True

if graficar:
    show = []

    for i in range(len(results)):
        show.append(map(lambda row: row['epoca'], results[i]))
        show.append(map(lambda row: row['funcion_de_costo'], results[i]))

    plt.plot(*show, marker='o')
    plt.xlabel('Epocas')
    plt.ylabel('Error/Funcion Costo')
    plt.show()

#plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epocas')
#plt.ylabel('Clasificaciones erroneas')
#plt.show()