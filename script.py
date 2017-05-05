# -*- coding: utf-8 -*-
import argparse
import os, sys
import parser as psr
import perceptron as ppn
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

def iniciar():
    usage = 'Este script tiene un único parametro obligatorio, que es el numero de ejercicio del TP. Puede ser 1 o 2 \n' \
          'Todos los demás son opcionales.\n' \
          'Ejemplo de ejecución: \n' \
          '$ python script.py 1 -ep=10000 -eta=0.01 -tr=20 -te=30 -val=50 -fa=tangente -dp=normal -tambatch=1 -mo=0'

    parser = argparse.ArgumentParser(usage=usage)

    # Argumento obligatorio: archivo de entrada
    parser.add_argument("nro_ejercicio", type=str, help='Numero de ejercicio. Valores: 1/2')

    # Argumentos opcionales: cantidad de epocas, eta, y proporcion de los datos
    # para usar como entrenamiento, test y validacion
    parser.add_argument("-ep", "--epochs", default=5000, help='Cantidad de epocas. Default = 5000')
    parser.add_argument("-eta", "--eta", default=0.01, help='Tasa de aprendizaje. Default = 0.01')
    parser.add_argument("-tr", "--train", default=33.33, help='% de input a utilizar como training. Default = 33')
    parser.add_argument("-te", "--test", default=33.33, help='% de input a utilizar como testing. Default = 33')
    parser.add_argument("-val", "--validation", default=33.33, help='% de input a utilizar como validation. Default = 33')
    parser.add_argument("-fa", "--factivacion", default='tangente',
                        help='Funcion de activacion a utilizar. Valores: tangente, logistica, tangente_optimizada')

    parser.add_argument("-dp", "--dpesos", default='normal',
                        help='Distribucion de pesos a utilizar. Valores: normal, uniforme')

    parser.add_argument("-tambatch", "--tambatch", default=1,
                        help='Tamanio del batch a utilizar')

    parser.add_argument("-mo", "--momentum", default=0,
                        help='Momentum a utilizar')

    args = parser.parse_args()

    nro_ejercicio = args.nro_ejercicio
    eta = float(args.eta)
    epochs = int(args.epochs)
    train_pct = float(args.train)
    test_pct = float(args.test)
    validation_pct = float(args.validation)
    f_activacion = args.factivacion
    d_pesos = args.dpesos
    tambatch = int(args.tambatch)
    momentum = float(args.momentum)

    os.system('clear')
    print 'TP1 - Perceptrón Multicapa'
    print "Se intentará procesar los datos del ejercicio "+nro_ejercicio+" ejecutando "+str(epochs)+" épocas con ETA "+str(eta)
    print str(train_pct) + "% del input utilizado como Entrenamiento"
    print str(test_pct) + "% del input utilizado como Testing"
    print str(validation_pct) + "% del input utilizado como Validacion"
    print "Funcion de activacion: " + f_activacion
    print "Distribucion de pesos: " + d_pesos
    print "Tamanio de batch: " + str(tambatch)
    print "Momentum: " + str(momentum)
    print '-------------------------------------------------------------------------'

    return nro_ejercicio, eta, epochs, train_pct, test_pct, validation_pct, f_activacion, d_pesos, tambatch, momentum

######### INICIO SCRIPT ##############

# Ejemplo de ejecucion:

nro_ejercicio, eta, epochs, train_pct, test_pct, validation_pct, \
    f_activacion, d_pesos, tambatch, momentum = iniciar()

i = psr.Parser()

datos_train, datos_validation, datos_test = i.parse(nro_ejercicio, train_pct, test_pct, validation_pct)

# Ejemplo de train
DATOS = datos_train

N_ENTRADA = len(DATOS[0][0]) - 1
RESULTADOS_ESPERADOS = [row[-1] for row in DATOS]

PPN = ppn.PerceptronMulticapa(N_ENTRADA, [3], 1, funcion_activacion=f_activacion,
                              distribucion_pesos=d_pesos, momentum=momentum)

results = PPN.train([row[0] for row in DATOS], RESULTADOS_ESPERADOS, eta=eta, epochs=epochs,
          tamanio_muestra_batch=tambatch)

DATOS_PREDICCION = [row[0] for row in DATOS]

resultados = []
esperados = []
for _ in range(100):
    for fila in DATOS_PREDICCION:
        prediccion = PPN.predecir(fila)
        resultados.append(prediccion)

    esperado = map(lambda row: row[-1], DATOS_PREDICCION)
    esperados = esperados + esperado

print "Eficiencia: %.2f %%" % PPN.medir_performance(esperados, resultados)

graficar = True

if graficar:
    # show = []

    epocas = map(lambda row: row['epoca'], results)
    errores = map(lambda row: row['funcion_de_costo'], results)

    # for i in range(len(results)):
    #     show.append(map(lambda row: row['epoca'], results[i]))
    #     show.append(map(lambda row: row['funcion_de_costo'], results[i]))

    # plt.plot(*show)

    plt.plot(epocas, errores, marker='o')
    plt.xlabel('Epocas')
    plt.ylabel('Error/Funcion Costo')
    plt.show()

#plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epocas')
#plt.ylabel('Clasificaciones erroneas')
#plt.show()