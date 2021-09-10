# import cv2 as cv
# import numpy as np
# import math
import csv
#
# from src.detection_main import apply_color_convertion, threshold, denoise, get_contours


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
    with open('../dataset/supervision.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row[0])

read_supervision()
# def get_hub_moments(file):
#     image = cv.imread(file)
#     gray = apply_color_convertion(frame=image, color=cv.COLOR_BGR2GRAY)
#     _, threshold_image = threshold(frame=gray, slider_max=255, trackbar_value=50)
#     denoised_image = denoise(frame=threshold_image, method=cv.MORPH_ELLIPSE, radius=5)
#     countours = get_contours(frame=denoised_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
#
#     moments = cv.moments(countours)  # momentos de inercia
#     # Calculate Hu Moments
#     huMoments = cv.HuMoments(moments)  # momentos de Hu
#
#     # Log scale hu moments
#     for i in range(0, 7):
#         huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(
#             abs(huMoments[i]))  # Mapeo para agrandar la escala.
#     return huMoments
