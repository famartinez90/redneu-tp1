# -*- coding: utf-8 -*-
import perceptron as ppn
import io, json
from json import JSONEncoder

def to_json(filepath, perceptron):
    with io.open(filepath, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(perceptron.__dict__, ensure_ascii=False)))

def from_json(filepath):
    with open(filepath, 'r') as content_file:
        content = content_file.read()
        json_object = json.loads(content)

    return ppn.PerceptronMulticapa(None, None,
                                   None, funcion_activacion=json_object['activacion_elegida'],
                                   distribucion_pesos=json_object['distribucion'],
                                   momentum=json_object['momentum'],
                                   basic_init=True, basic_init_pesos=json_object['pesos_red'])
