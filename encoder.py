# -*- coding: utf-8 -*-
import perceptron as ppn
import io, json

DATOS = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0],
         [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0],
         [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
         [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1],
         [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]

N_ENTRADA = len(DATOS[0]) - 1
RESULTADOS_ESPERADOS = [xs[-1] for xs in DATOS]

PPN = ppn.PerceptronMulticapa(N_ENTRADA, [3], 1, funcion_activacion="tangente", distribucion_pesos="normal", momentum=0)
PPN.train([xs[:-1] for xs in DATOS], RESULTADOS_ESPERADOS, eta=0.05, epochs=2, tamanio_muestra_batch=1, adaptativo=False)

with io.open('perceptron1.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps(PPN, ensure_ascii=False))