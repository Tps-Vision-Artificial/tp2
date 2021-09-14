import csv


def label_to_int(string_label):
    if string_label == '5-point-star': return 1
    if string_label == 'rectangle': return 2
    if string_label == 'triangle':
        return 3

    else:
        raise Exception('unkown class_label')

def int_to_label(string_label):

    if string_label == 0: return '5-point-star'
    if string_label == 1: return 'rectangle'
    if string_label == 2:
        return 'triangle'
    else:
        raise Exception('unkown class_label')

def read_supervision():
    with open('./dataset/supervision.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            print('image: ' + row[0] + ' shape: ' + row[1])

def read_tags():
    with open('./dataset/tags.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            print('index: ' + row[0] + ' label: ' + row[1])


def int_to_label(int_value):
    with open('./dataset/tags.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "\ufeff0":
                row[0] = 0
            if int(row[0]) == int_value:
                return row[1]
