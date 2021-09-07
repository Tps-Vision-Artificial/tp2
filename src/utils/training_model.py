import cv2 as cv
import numpy as np
import csv

from src.utils.label_converters import label_to_int

train_data = []
train_label = []

def load_training_set():
    global trainData
    global trainLabels
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop() # saca el ultimo elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)
# transforma los arrays a arrays de forma numpy


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    load_training_set()

    tree = cv.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)
    return tree
