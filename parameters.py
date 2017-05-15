# -*- coding: utf-8 -*-
import argparse
import os

def iniciar():
    usage = 'Este script tiene un único parametro obligatorio, que es el numero de ejercicio del TP. Puede ser 1 o 2 \n' \
          'Todos los demás son opcionales.\n' \
          'Ejemplo de ejecución: \n' \
          '$ python script.py 1 -ep=10000 -eta=0.01 -tr=20 -te=30 -val=50 -fa=tangente -dp=normal -tambatch=1 -mo=0'

    parser = argparse.ArgumentParser(usage=usage)

    # Argumento obligatorio: ejercicio a resolver
    parser.add_argument("nro_ejercicio", type=str, help='Numero de ejercicio. Valores: 1/2')

    # Argumentos opcionales:

    parser.add_argument("-file", "--filepath", default=None, help='Ubicacion del archivo con los datasets a procesar')
    parser.add_argument("-ep", "--epochs", default=500, help='Cantidad de epocas. Default = 500')
    parser.add_argument("-eta", "--eta", default=0.05, help='Tasa de aprendizaje. Default = 0.05')
    parser.add_argument("-capas", "--capas", default='10,10',
                        help='Cantidad de capas ocultas y neuronas en cada una. Se representa como una secuencia '
                             'de enteros separados por coma.'
                             'Donde cada elemento i es el número de neuronas en la capa i. '
                             ' Ejemplo: 2,4 sera una red de 2 capas, la primera de 2 y la segunda de 4'
                             'neuronas.'
                             'La longitud de elementos es la cantidad de capas ocultas. Default = 10,10. 2 capas de 10 neuronas.')

    parser.add_argument("-tr", "--train", default=70, help='% de input a utilizar como training. Default = 70')
    parser.add_argument("-te", "--test", default=20, help='% de input a utilizar como testing. Default = 20')
    parser.add_argument("-val", "--validation", default=10, help='% de input a utilizar como validation. Default = 10')
    parser.add_argument("-fa", "--factivacion", default='tangente',
                        help='Funcion de activacion a utilizar. Valores: tangente, logistica, tangente_optimizada')

    parser.add_argument("-dp", "--dpesos", default='normal',
                        help='Distribucion de pesos a utilizar. Valores: normal, uniforme')

    parser.add_argument("-tambatch", "--tambatch", default=1,
                        help='Tamanio del batch a utilizar')

    parser.add_argument("-mo", "--momentum", default=0,
                        help='Momentum a utilizar')

    parser.add_argument("-rda", "--red_desde_archivo", default=None,
                        help='Permite elegir una red ya entrenada. Las redes estan almacenadas en archivos.'
                             'Este parametro toma un filepath que contenga un txt con una red. Opciones: red_ej1.txt, red_ej2.txt')

    parser.add_argument("-rha", "--red_hacia_archivo", default=None,
                        help='Permite elegir una red ya entrenada. Las redes estan almacenadas en archivos json.'
                             'Este parametro toma un filepath que contenga una red en formato json. Opciones: red_ej1.json, red_ej2.json')

    parser.add_argument("-estop", "--earlystopping", default=0,
                        help='Treshold para hacer el early stopping')

    parser.add_argument("-adap", "--adaptativo", default=0,
                        help='Ejecutar con o sin parámetros adaptativos. Valores = 0 / 1')

    args = parser.parse_args()

    nro_ejercicio = args.nro_ejercicio
    filepath = args.filepath

    eta = float(args.eta)
    epochs = int(args.epochs)
    capas = args.capas
    capas_list = capas.split(",")
    capas_list = map(int, capas_list)

    train_pct = float(args.train)
    test_pct = float(args.test)
    validation_pct = float(args.validation)
    f_activacion = args.factivacion
    d_pesos = args.dpesos
    tambatch = int(args.tambatch)
    momentum = float(args.momentum)
    red_desde_archivo = args.red_desde_archivo
    red_hacia_archivo = args.red_hacia_archivo

    estop = float(args.earlystopping)
    adaptativo = bool(args.adaptativo)

    os.system('clear')
    print 'TP1 - Perceptrón Multicapa'
    print "Se intentará procesar los datos del ejercicio "+nro_ejercicio+" ejecutando "+str(epochs)+" épocas con ETA "+str(eta)
    print str(train_pct) + "% del input utilizado como Entrenamiento"
    print str(test_pct) + "% del input utilizado como Testing"
    print str(validation_pct) + "% del input utilizado como Validacion"
    print "Capas ocultas: " + str(capas)
    print "Funcion de activacion: " + f_activacion
    print "Distribucion de pesos: " + d_pesos
    print "Tamanio de batch: " + str(tambatch)
    print "Momentum: " + str(momentum)
    print "Early Stopping: " + str(estop)
    print "Parámetros Adaptativos: " + str(adaptativo)
    print "Red a Utilizar: " + (red_desde_archivo if (red_desde_archivo is not None) else 'Nueva')
    print '-------------------------------------------------------------------------'

    return nro_ejercicio, filepath, eta, epochs, capas_list, train_pct, test_pct, validation_pct, \
           f_activacion, d_pesos, tambatch, momentum, red_desde_archivo, red_hacia_archivo, estop, adaptativo
