import perceptron as ppn
import matplotlib.pyplot as plt

# Ejemplo de train
DATOS = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], 
         [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0], 
         [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1], 
         [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], 
         [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]

N_ENTRADA = len(DATOS[0]) - 1
RESULTADOS_ESPERADOS = [xs[-1] for xs in DATOS]

results = []
for _ in range(10):
    PPN = ppn.PerceptronMulticapa(N_ENTRADA, [3], 1, funcion_activacion="tangente", distribucion_pesos="normal", momentum=0)
    results.append(PPN.train([xs[:-1] for xs in DATOS], RESULTADOS_ESPERADOS, eta=0.05, epochs=70, tamanio_muestra_batch=1, adaptativo=False))

DATOS_PREDICCION = [[2.7810836, 2.550537003, 0],
                    [1.465489372, 2.362125076, 0],
                    [3.396561688, 4.400293529, 0],
                    [1.38807019, 1.850220317, 0],
                    [3.06407232, 3.005305973, 0],
                    [7.627531214, 2.759262235, 1],
                    [5.332441248, 2.088626775, 1],
                    [6.922596716, 1.77106367, 1],
                    [8.675418651, -0.242068655, 1],
                    [7.673756466, 3.508563011, 1]]

resultados = []
esperados = []
for _ in range(100):
    for fila in DATOS_PREDICCION:
        prediccion = PPN.predecir_test(fila)
        resultados.append(prediccion)
    
    esperado = map(lambda xs: xs[-1], DATOS_PREDICCION)
    esperados = esperados + esperado

print "Eficiencia: %.2f %%" % PPN.medir_performance(esperados, resultados)

# print PPN.propagacion_forward([17.673756466, 13.508563011, 1])

# Se grafica la cantidad de clasificaciones incorrectas en cada epoca versus numero de epoca. 
# MEJORAS SOLICITADAS: Mejora 1) Graficar la funcion de costo versus el numero de epoca.
# Mejora 2) Agregar al grafico el error de costo del conjunto de validacion (o sea, debe graficar en el mismo grafico, ambos errores de costo)

graficar = True

if graficar:
    show = []

    for i in range(len(results)):
        show.append(map(lambda row: row['epoca'], results[i]))
        show.append(map(lambda row: row['funcion_de_costo'], results[i]))

    plt.plot(*show)
    plt.xlabel('Epocas')
    plt.ylabel('Error/Funcion Costo')
    plt.show()

# MEJORAS SOLICITADAS: Agregar imprimir la funcion de costo y el root-mean-square error final del conjunto de testing, 
# para saber la performance final de la red neuronal.

# NOTE: High momentum should always be accompanied by low learning rate, else you will overshoot the global optimum.