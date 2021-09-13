import os
import cv2 as cv
import numpy as np
from math import log10, copysign
import csv


# def generate_hu_moments_file():
#
#     with open('generated-files/shapes-hu-moments.csv', 'w',
#               newline='') as file:  # Se genera un archivo nuevo (W=Write)
#         writer = csv.writer(file)
#         # Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
#         # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
#         write_hu_moments(0, writer)
#         write_hu_moments(1, writer)
#         write_hu_moments(2, writer)
#
#
def get_pictures(myDir ,format='.png'):
    fileList = []
    for root,dirs,files in os.walk(myDir, topdown = False):
        for name in files:
            fullname = os.path.join(root, name)
            fileList.append(fullname)
    return fileList

def read_supervision():
    descriptors = []
    pictures = get_pictures('../dataset/images')
    with open('../dataset/supervision.csv') as file:
        reader = csv.reader(file)
        position = 0
        for row in reader:
            hu_moments = [get_hub_moments(pictures[position])]
            element = [row[0], hu_moments]
            descriptors.append(element)
            position = position + 1

    with open('../dataset/descriptores.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(descriptors)


def get_hub_moments(file):
    image = cv.imread(file)
    gray = apply_color_convertion(image, cv.COLOR_BGR2GRAY)
    _, threshold_image = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    denoised_image = denoise(frame=threshold_image, method=cv.MORPH_ELLIPSE, radius=5)
    countours = get_contours(frame=denoised_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    moments = cv.moments(countours[1])  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv.HuMoments(moments)  # momentos de Hu

    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments


def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing


def get_contours(frame, mode, method):
    contours, hierarchy = cv.findContours(frame, mode, method)
    return contours


def apply_color_convertion(frame, color):
    return cv.cvtColor(frame, color)


read_supervision()
