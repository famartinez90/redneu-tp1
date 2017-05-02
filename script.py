# -*- coding: utf-8 -*-
import argparse
import os
import parser as psr
import perceptron as ppn
import matplotlib.pyplot as plt

def iniciar():
    usage = 'Este script tiene un único parametro obligatorio, que es el path del archivo de entrada con los datos de input \n' \
          'Todos los demás son opcionales.\n' \
          'Ejemplo de ejecución: \n' \
          '$ python perceptron.py tp1_ej1_training.csv -ep=10000 -eta=0.01 -tr=20 -te=30 -val=50'

    parser = argparse.ArgumentParser(usage=usage)

    # Argumento obligatorio: archivo de entrada
    parser.add_argument("input_file", type=str, help='Path del archivo con datos de entrada')

    # Argumentos opcionales: cantidad de epocas, eta, y proporcion de los datos
    # para usar como entrenamiento, test y validacion
    parser.add_argument("-ep", "--epochs", default=50000, help='Cantidad de epocas. Default = 50.000')
    parser.add_argument("-eta", "--eta", default=0.01, help='Tasa de aprendizaje. Default = 0.01')
    parser.add_argument("-tr", "--train", default=33.33, help='% de input a utilizar como training. Default = 33')
    parser.add_argument("-te", "--test", default=33.33, help='% de input a utilizar como testing. Default = 33')
    parser.add_argument("-val", "--validation", default=33.33, help='% de input a utilizar como validation. Default = 33')

    args = parser.parse_args()

    input_file = args.input_file
    eta = float(args.eta)
    epochs = int(args.epochs)
    train_pct = float(args.train)
    test_pct = float(args.test)
    validation_pct = float(args.validation)

    os.system('clear')
    print 'TP1 - Perceptrón Multicapa'
    print "Se intentará procesar los datos en "+input_file+" ejecutando "+str(epochs)+" épocas con ETA "+str(eta)
    print str(train_pct) + "% del input utilizado como Entrenamiento"
    print str(test_pct) + "% del input utilizado como Testing"
    print str(validation_pct) + "% del input utilizado como Validacion"
    print '-------------------------------------------------------------------------'

    return input_file, eta, epochs, train_pct, test_pct, validation_pct

######### INICIO SCRIPT ##############

# Ejemplo de ejecucion:

input_file, eta, epochs, train_pct, test_pct, validation_pct = iniciar()

i = psr.Parser()
datos_train, datos_validation, datos_test = i.parse(input_file, train_pct, test_pct, validation_pct)

# Ejemplo de train
DATOS = datos_train

N_ENTRADA = len(DATOS[0]) - 1
N_SALIDA = len(set([row[0] for row in DATOS]))

PPN = ppn.PerceptronMulticapa(N_ENTRADA, [3], 2, funcion_activacion="logistica", distribucion_pesos="normal", momentum=0)
PPN.train_batch(DATOS, N_SALIDA, eta=eta, epochs=epochs, tamanio_muestra_batch=1)

DATOS_PREDICCION = datos_validation

resultados = []
esperados = []
for _ in range(epochs):
    for fila in DATOS_PREDICCION:
        prediccion = PPN.predecir(fila)
        resultados.append(prediccion)
    
    esperado = map(lambda xs: xs[-1], DATOS_PREDICCION)
    esperados = esperados + esperado

print "Eficiencia: %.2f %%" % PPN.medir_performance(esperados, resultados)

print PPN.propagacion_forward([17.673756466, 13.508563011, 1])

#plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
#plt.xlabel('Epocas')
#plt.ylabel('Clasificaciones erroneas')
#plt.show()