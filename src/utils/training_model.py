import cv2 as cv
import numpy as np
import csv


trainData = []
trainLabels = []


def load_training_set():
    global trainData
    global trainLabels
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop(0)  # saca el primer elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n))  # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32))  # momentos de Hu
            trainLabels.append(np.array([float(class_label)], dtype=np.int32))  # Resultados
            # Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)
    print(trainData)
    print(trainLabels)


# transforma los arrays a arrays de forma numpy


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    load_training_set()

    tree = cv.ml.DTrees_create()
    tree.setCVFolds(1)
    # depth => number of decisions to take
    # try to decrease depth. maybe works better
    tree.setMaxDepth(10)
    tree.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)
    tree.save('./generated-files/trained.yml')
    return tree
