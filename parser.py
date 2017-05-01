# -*- coding: utf-8 -*-
import csv
import random

class Parser(object):

    def parse(self, filepath, train_pct, test_pct, validation_pct):
        try:
            reader = csv.reader(open(filepath, 'r'))
            cant_datos = len(open(filepath).readlines())

            # Estimo cantidad de registros que ira a cada set de datos
            cant_datos_training = int(cant_datos * train_pct / 100)
            cant_datos_test = int(cant_datos * test_pct / 100)
            cant_datos_validation = int(cant_datos * validation_pct / 100)

            datos = []
            datos_train = []
            datos_test = []
            datos_validation = []

            # Paso CSV a lista de rows para manejar mas facil
            for row in reader:
                datos.append(row)

            # Mezclo datos para que la seleccion y division sean azarosas
            random.shuffle(datos)

            i = 0
            for row in datos:
            # Divido los datos del input en porciones respectivas de training, test y validation
                index_training = (0, cant_datos_training)
                index_test = (index_training[1], index_training[1] + cant_datos_test)
                index_validation = (index_test[1], index_test[1] + cant_datos_validation)

                if i in range(*index_training):
                    datos_train.append(row)
                elif i in range(*index_test):
                    datos_test.append(row)
                elif i in range(*index_validation):
                    datos_validation.append(row)

                i += 1

            return datos_train, datos_validation, datos_test

        except Exception as e:
            print "Error al parsear archivo de entrada"
            print str(e)