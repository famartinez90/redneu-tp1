# -*- coding: utf-8 -*-
import argparse
import os
import parser as psr
import perceptron as ppn

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
    eta = args.eta
    epochs = args.epochs
    train_pct = args.train
    test_pct = args.test
    validation_pct = args.validation

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

# TODO: conectar datos del parser al perceptron multicapa