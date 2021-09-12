import cv2
import numpy as np
import math
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
# def write_hu_moments(files, writer, label):
#     hu_moments = []
#     for file in files:
#         hu_moments.append(get_hub_moments(file))
#     for mom in hu_moments:
#         flattened = mom.ravel()
#         row = np.append(flattened, label)
#         writer.writerow(row)

def read_supervision():
    descriptores = []
    with open('../dataset/supervision.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            element = [row[0], get_hub_moments('../dataset/imagenes/'+row[0])]
            descriptores.append(element)

    with open('../dataset/descriptores.csv','w', newline='') as file:
        writer = csv.writer(file,delimiter =',')
        writer.writerows(descriptores)

def get_hub_moments(file):
     image = cv2.imread(file)
     gray = apply_color_convertion(image,cv2.COLOR_BGR2GRAY)
     _, threshold_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
     denoised_image = denoise(frame=threshold_image, method=cv2.MORPH_ELLIPSE, radius=5)
     countours = get_contours(frame=denoised_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

     moments = cv2.moments(countours)  # momentos de inercia
     # Calculate Hu Moments
     huMoments = cv2.HuMoments(moments)  # momentos de Hu

     # Log scale hu moments
     for i in range(0, 7):
         huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(
             abs(huMoments[i]))  # Mapeo para agrandar la escala.
     return huMoments

def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours

def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)

read_supervision()