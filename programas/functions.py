#!/usr/bin/env python
# coding: utf-8

import nltk
import numpy as np
from fuzzywuzzy import fuzz
import re


def cleaner(w):
    vector = np.array([[r'\'e1', r'á'],
                       [r'\'e9', r'é'],
                       [r'\'ed', r'í'],
                       [r'\'cd', r'í'],
                       [r'\'fa', r'ú'],
                       [r'\'f3', r'ó'],
                       [r'\'e0', r'à'],
                       [r'\'e8', r'è'],
                       [r'\'ec', r'ì'],
                       [r'\'f2', r'ò'],
                       [r'\'f9', r'ù'],
                       [r'\'c1', r'Á'],
                       [r'\'d3', r'Ó'],
                       [r'\tab ', ''],
                       [r'\n', ''],
                       [r'\'d1', r'ñ'],
                       [r'\par_x000D_', ''],
                       [r'_x000D_', '']])

    w = w.lower()
    w = w.replace('descripcion hecho', '')
    w = w.split()
    for i, desc in enumerate(w):
        for row in vector:
            if row[0] in desc:
                desc.replace(row[0], row[1])
        w[i] = desc
    return ' '.join(w)


def punt(w):
    w = re.sub('[^\w\s]', '', w)
    w = re.sub('(\r\n?|\n)+', '', w)
    w = re.sub('\d+', '', w)
    w = re.sub('\d', '', w)
    w = w.replace('\'da', 'u')

    return w


def nonStop(w):
    return ' '.join(i for i in w.split() if i not in ['el', 'la', 'los', 'las', 'ellos', 'nosotros', 'lo', 'le', 'que'])


def separador(ds):
    ds_a = ds[ds['cod_accidente'] == 'AA    ']
    ds_m = ds[ds['cod_accidente'] == 'AM    ']
    ds_b = ds[ds['cod_accidente'] == 'ACI   ']
    ds_p = ds[ds['cod_accidente'] == 'PEATON']
    return ds_a, ds_m, ds_b, ds_p


def conIdxPart(w):
    part = ['parte', 'lateral', 'costado']
    idx = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(w))
    aux = ''
    for j in part:
        for i in idx.find_concordance(j):
            aux += ' '.join(i[0][-3:]) + ' ' + i[1] + ' ' + ' '.join(i[2][:3])
            aux += ', '
    return aux


def conIdxColis(w):
    colis = ['colisionada', 'colisiona', 'colisionado', 'colisionando', 'colisiono', 'colision', 'impactarlo',
             'impactoel', 'impactarla', 'impactado', 'impacta', 'impacte', 'impactada', 'impactar', 'impactando',
             'impactandolo', 'impacto']
    idx = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(w))
    aux = ''
    for j in colis:
        for i in idx.find_concordance(j):
            aux += ' '.join(i[0][-5:]) + ' ' + i[1] + ' ' + ' '.join(i[2][:5])
            aux += ', '
    return aux

# las 4 funciones siguientes se deben simplificar para cada clave que busquemos


def PosParte(i):
    pos = []
    for idx, j in enumerate(i):
        if j.text == 'parte':
            try:
                pos.append([i[x].pos_ for x in range(idx - 5, idx + 6)])
            except IndexError:
                pos.append([i[x].pos_ for x in range(idx - 5, len(i))])
    return pos


def LemParte(i):
    lemma = []
    for idx, j in enumerate(i):
        if j.text == 'parte':
            try:
                lemma.append([i[x].lemma_ for x in range(idx - 5, idx + 6)])
            except IndexError:
                lemma.append([i[x].lemma_ for x in range(idx - 5, len(i))])
    return lemma


def PosColis(i):
    colis = ['colisionada', 'colisiona', 'colisionado', 'colisionando', 'colisiono', 'colision', 'impactarlo',
             'impactoel', 'impactarla', 'impactado', 'impacta', 'impacte', 'impactada', 'impactar', 'impactando',
             'impactandolo', 'impacto']
    pos = []
    # lemma=[]
    for idx, j in enumerate(i):
        if j.text in colis:
            try:
                pos.append([i[x].pos_ for x in range(idx - 5, idx + 6)])
            except IndexError:
                pos.append([i[x].pos_ for x in range(idx - 5, len(i))])
    return pos


def LemColis(i):
    colis = ['colisionada', 'colisiona', 'colisionado', 'colisionando', 'colisiono', 'colision', 'impactarlo',
             'impactoel', 'impactarla', 'impactado', 'impacta', 'impacte', 'impactada', 'impactar', 'impactando',
             'impactandolo', 'impacto']

    lemma = []
    for idx, j in enumerate(i):
        if j.text in colis:
            try:
                lemma.append([i[x].lemma_ for x in range(idx - 5, idx + 6)])
            except IndexError:
                lemma.append([i[x].lemma_ for x in range(idx - 5, len(i))])
    return lemma


def pari2part(w):
    aux = []
    if type(w) == list:
        for x in w:
            aux.append([i.replace('partir', 'parte') for i in x])
        return aux
    return [i.replace('partir', 'parte') for i in w]


def categorizador(w):
    dic = ['izquierda', 'izquierdo', 'derecha', 'derecho', 'delantera', 'delantero', 'trasero', 'trasera']
    label = np.array([0, 0, 0, 0])
    for word in w.split():
        if word == dic[0] or word == dic[1] or word.startswith('izq'):
            label[0] += 1
        elif word == dic[2] or word == dic[3]:
            label[1] += 1
        elif word == dic[4] or word == dic[5] or word.startswith('frent') or word.startswith('front'):
            label[2] += 1
        elif word == dic[6] or word == dic[7]:
            label[3] += 1
    return [i for i in range(len(label)) if label[i] == label.max()][0]

# Devuelve la palabra más cercana según el algoritmo de levenshtein


def max_ratio(w):
    try:
        dic = ['parte', 'circulaba', 'delantera', 'delantero', 'delante', 'enfrente', 'izquierda', 'izquierdo',
               'derecha', 'lateral', 'colisiona', 'colision', 'colisiono', 'colisionada', 'colisionado', 'colisionando',
               'trasero', 'trasera',
               'impacta', 'impacto', 'impactado', 'impactada', 'embiste', 'embistio', 'embestido', 'iba', 'venia',
               'frente', 'lado', 'detras', 'atras', 'costado', 'choca', 'choco', 'chocando', 'choque', 'costado',
               'frontal', 'asegurado', 'cae', 'venia', 'piso', 'aseg', 'lesiones', 'medios', 'vehiculo', 'ocupante',
               'persona',
               'datos', 'propios', 'desplazamientos', 'solo', 'llegar', 'puerta', 'cayo', 'ultima', 'hecho', 'caen',
               'maniobra', 'acompanante', 'segun', 'pavimento', 'hospital', 'espejo', 'ambos', 'habia', 'suelo',
               'tenia', 'frena', 'mecanica', 'levanta', 'ocupantes', 'momento', 'dolor', 'velocidad', 'version',
               'personas', 'san', 'asfalto', 'marcha', 'llevaba', 'retira', 'mismo', 'sola', 'produce', 'ingresar',
               'puesto', 'trasladado', 'luz', 'presentaba', 'retiro', 'maniobro', 'tomar', 'asfalto']

        aux = 0
        word = ''
        for i in dic:
            if (aux <= fuzz.ratio(w, i) and 90 <= fuzz.ratio(w, i)):  # or 95 <= fuzz.partial_ratio(w, i):
                aux = max(fuzz.ratio(w, i), fuzz.partial_ratio(w, i))
                word = i
        # print(w,word)
        if word != '':
            return word
        return w
        # return w hay que poner para usar, mientras me sirve para evaluar
    except TypeError:
        return w

# Retorna el string que contiene la palabra 'aseg' y derivados


def Aseg(w):
    c = 0
    for i in w.split():
        if i.startswith('aseg'):
            c += 1
    if c == 0:
        return w
    return ''

# to_find es la terna de pod_tags a encontrar, where es la columna donde vas a buscar
# where es pd.Series y cada elemento es spacy.doc.Doc
# retorna una lista con las ternas encontradas.


def findStruct(to_find, where):
    words = []
    for i in where:
        for j in range(len(i)):
            if i[j].pos_ == to_find[0] and j + 2 < len(i):

                if i[j + 1].pos_ == to_find[1]:

                    if i[j + 2].pos_ == to_find[2]:
                        words.append(i[j].text + ' ' + i[j + 1].text + ' ' + i[j + 2].text)
    return words

# Etiquetas básicas para clasificar


def Labeler():
    l1 = ['parte', 'lateral']
    l2 = ['izquierda', 'derecha', 'delantera', 'trasera']
    l3 = ['izquierdo', 'derecho', 'delantero', 'trasero']
    target = []
    for i in l2:
        target.append(l1[0] + ' ' + i)
    for i in l3:
        target.append(l1[1] + ' ' + i)

    target += ['izquierda', 'derecha', 'adelante', 'atras']
    return target


def GeoCenter(setvecs):
    suma_x, suma_y = 0, 0
    for i in setvecs:
        suma_x += i[0]
        suma_y += i[1]
    return np.array([suma_x / len(setvecs), suma_y / len(setvecs)])
