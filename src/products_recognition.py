
import time as t
import argparse

import glob
import os
import ast

from PIL import Image

import pandas as pd

import clip
import ocr
from PaddleOCR import paddleocr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


def matriz_confusion(diccionario, clip=True):
    # Paso 1: crear listas de valores reales y predicciones
    valores_reales = []
    predicciones = []
    for valor_real, lista_predicciones in diccionario.items():
        valores_reales.extend([valor_real] * len(lista_predicciones))
        predicciones.extend(lista_predicciones)

    # Paso 2: obtener todas las etiquetas
    etiquetas = list(set(valores_reales + predicciones))
    etiquetas.sort()
    # Paso 3: crear la matriz de confusión
    matriz = [[0 for _ in etiquetas] for _ in etiquetas]

    # Paso 4: recorrer las listas y actualizar la matriz de confusión
    for valor_real, pred in zip(valores_reales, predicciones):
        i = etiquetas.index(valor_real)
        j = etiquetas.index(pred)
        matriz[i][j] += 1
    matriz = pd.DataFrame(matriz, columns=etiquetas, index=etiquetas)
    # csv y xlsx
    ruta_archivo = 'matriz_confusion_clip.csv'
    matriz.to_csv(ruta_archivo, index=False)
    ruta_archivo = 'matriz_confusion_clip.xlsx'
    matriz.to_excel(ruta_archivo, index=False)

    # Calcular el tamaño de la figura
    fig_width = 0.5 * len(matriz.columns)
    fig_height = 0.5 * len(matriz.index)

    # Ajustar el tamaño de la figura
    plt.figure(figsize=(fig_width, fig_height))

    # Crear el mapa de calor con seaborn
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')

    # Obtener los ejes del mapa de calor
    ax = plt.gca()

    # Mostrar todos los dígitos en los ejes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       ha='center', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       ha='right', fontsize=8)
    # put the name of the axes in a adoc size

    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')

    # set the size of the axes names in adoc size
    ax.xaxis.label.set_size(32)
    ax.yaxis.label.set_size(32)
    # set title
    ax.set_title("Matriz de Confusión")
    ax.title.set_size(64)

    # Mostrar el mapa de calor

    if clip:
        plt.savefig('matriz_clip.png', dpi=300)
    else:
        plt.savefig('matriz_ocr.png', dpi=300)
    return matriz


def matriz_confusion_comprimida(diccionario, clip=True):
    # Crear la matriz de confusión comprimida
    matriz_ = [[0, 0], [0, 0]]  # [TP, FP], [FN, TN]

    # Recorrer el diccionario y actualizar la matriz de confusión comprimida
    for valor_real, predicciones in diccionario.items():
        tp_count = predicciones.count(valor_real)
        fn_count = len(predicciones) - tp_count

        matriz_[0][0] += tp_count  # Verdadero positivo (TP)
        matriz_[0][1] += fn_count  # Falso positivo (FP)
        matriz_[1][0] += fn_count  # Falso negativo (FN)
        matriz_[1][1] += tp_count  # Verdadero positivo (TP)

    # Convertir la matriz de confusión comprimida a un DataFrame
    etiquetas = ["medicamentos", "no medicamentos"]
    matriz = pd.DataFrame(matriz_, columns=etiquetas, index=etiquetas)

    # Generar el gráfico de la matriz de confusión comprimida
    fig_width = 5
    fig_height = 5
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                       ha='center', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       ha='right', fontsize=8)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_title("Matriz de Confusión")
    plt.tight_layout()

    # Guardar el gráfico en un archivo
    if clip:
        plt.savefig('matriz_confusion_clip_comprimida.png', dpi=300)
    else:
        plt.savefig('matriz_confusion_ocr_comprimida.png', dpi=300)

    return matriz_


def generar_curvas_roc(matriz_confusion):
    # Calcular las curvas ROC
    tp = matriz_confusion[0][0]  # Verdaderos positivos
    fp = matriz_confusion[0][1]  # Falsos positivos
    fn = matriz_confusion[1][0]  # Falsos negativos
    tn = matriz_confusion[1][1]  # Verdaderos negativos

    tp_vector = np.ones(tp)
    tn_vector = np.zeros(tn)
    fp_vector = np.ones(fp)
    fn_vector = np.zeros(fn)
    # print("tp: ", tp_vector)
    # print("tn: ", tn_vector)
    # print("fp: ", fp_vector)
    # print("fn: ", fn_vector)

    y_true = np.concatenate((tp_vector, tn_vector, fn_vector, fp_vector))
    y_pred = np.concatenate(
        (np.ones(tp + fn), np.zeros(tn + fp)))
    # print(y_true)
    # print(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.clf()
    # print("tpr: ", tpr)
    # print("fpr: ", fpr)
    plt.plot(fpr, tpr, linestyle='--')
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend()
    plt.savefig('curva_roc.png', dpi=300)


def clip_recognition(csv_path, images_path, imgs, model_name):
    images = f'{images_path}M10-OAI-F.OAI FARMA/'
    df = pd.read_csv(csv_path)
    tiemop_carga = t.time()
    # clip.save_model(args.model)
    model, preprocess, device = clip.load_model(model_name)
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo CLIP: ", tiempo_carga_stop-tiemop_carga)

    resultados_clip = [(clip.inference(model, preprocess, device, img, df), int(
                        img.replace(images, "")[:6])) for img in imgs]
    correctos = 0
    totales = len(resultados_clip)
    tiempo_total = 0
    dict_resultados_clip = {
        int(img.replace(images, "")[:6]): [] for img in imgs}
    for result, sku in resultados_clip:
        dict_resultados_clip[sku].append(result[0])
        tiempo_total += result[2]
        if (result[0] == sku):
            correctos += 1
    matriz_clip = matriz_confusion(dict_resultados_clip, clip=True)
    matriz_comprimida = matriz_confusion_comprimida(
        dict_resultados_clip, clip=True)

    generar_curvas_roc(matriz_comprimida)
    etiquetas = list(dict_resultados_clip.keys())
    df = pd.DataFrame(matriz_clip, columns=etiquetas, index=etiquetas)
    ruta_archivo = 'matriz_confusion_clip.xlsx'
    df.to_excel(ruta_archivo, index=False)
    # grafico_matriz_confusion(matriz_clip, dict_resultados_clip)
    print("Correctos: ", correctos)
    print("Totales: ", totales)
    print("Porcentaje de acierto: ", 100*correctos/totales)
    print("Tiempo de inferencia: ", tiempo_total)
    print("Tiempo de inferencia promedio: ", tiempo_total/totales)
    print("\n")


def clip_single_recognition(dataframe, img, model):
    """Uso de clip que recibe una imagen y devuelve el resultado"""
    # clip.save_model(args.model)
    model, preprocess, device = model
    _, _, _, skus_pred, _ = clip.inference(
        model, preprocess, device, img, dataframe)
    return skus_pred[0]


def clip_filter(csv_path, clip_load, img, results=5):
    """Uso de clip que recibe todas las imagenes y devuelve la cantidad dada de resultados"""
    df = pd.read_csv(csv_path)

    # clip.save_model(args.model)
    model, preprocess, device = clip_load

    return clip.inference(model, preprocess, device, img, df)[3][:results]


def ocr_filter(img, dataframe, model, results=5):
    """Uso de ocr que recibe todas las imagenes y devuelve la cantidad dada de resultados"""

    words, _ = ocr.ocr_inference(model, img, 0)
    sku = int(img.split('/')[-2])

    skus = []
    for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
        sku = str(sku).zfill(6)
        text = ast.literal_eval(text)
        score = ocr.ocr_match_2(text, words, 0.8)
        if (score != 0):
            skus.append((sku, score))
    skus.sort(key=lambda tup: tup[1], reverse=True)
    # print(skus)
    if (len(skus) == 0):
        return 0
    filtro = [int(sku) for sku, _ in skus]
    return filtro[:results]


def ocr_single_recognition(img, dataframe, model):
    """with a img and df ocr do the match"""
    words, _ = ocr.ocr_inference(model, img, 0)
    sku = int(img.split('/')[-2])

    skus = []
    for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
        sku = str(sku).zfill(6)
        text = ast.literal_eval(text)
        score = ocr.ocr_match_2(text, words, 0.8)
        if (score != 0):
            skus.append((sku, score))
    skus.sort(key=lambda tup: tup[1], reverse=True)
    print(skus)
    if (len(skus) == 0):
        return 0
    return skus[0][0]


def ocr_recognition(images_path, imgs, dataframe):
    images = f'{images_path}M10-OAI-F.OAI FARMA/'

    dict_resultados_ocr = {
        int(img.replace(images, "")[:6]): [] for img in imgs}

    # # OCR
    tiemop_carga = t.time()
    engine = ocr.ocr_load_model()
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo OCR: ", tiempo_carga_stop-tiemop_carga)
    resultados_ocr = []
    for img in imgs:
        words, t_inference = ocr.ocr_inference(engine, img, 0)
        sku = int(img.replace(images, "")[:6])
        # print(f"Tiempo de inferencia: {t_inference}")
        resultados_ocr.append((words, t_inference, sku))

    tiempo_match = []
    correctos = 0
    for result, time, img in resultados_ocr:
        start_match = t.time()
        skus = []
        for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
            sku = str(sku).zfill(6)
            text = ast.literal_eval(text)
            score = ocr.ocr_match_2(text, result, 0.8)
            if (score != 0):
                skus.append((sku, score))
        skus.sort(key=lambda tup: tup[1], reverse=True)
        if (len(skus) > 0):
            sku = skus[0][0]
            dict_resultados_ocr[img].append(int(sku))
            if (int(sku) == int(img)):
                correctos += 1
                # print(f"Correcto: {sku} - {img}")
        end = t.time()
        # print("Tiempo de match: ", end-start_match)
        tiempo_match.append(end-start_match)
    matriz_ocr = matriz_confusion(dict_resultados_ocr, False)
    matriz_comprimida = matriz_confusion_comprimida(dict_resultados_ocr, False)
    generar_curvas_roc(matriz_comprimida)
    # grafico_matriz_confusion(matriz_ocr, dict_resultados_ocr, False)

    etiquetas = list(dict_resultados_ocr.keys())
    df = pd.DataFrame(matriz_ocr, columns=etiquetas, index=etiquetas)
    ruta_archivo = 'matriz_confusion_ocr.xlsx'
    df.to_excel(ruta_archivo, index=False)

    print("Correctos: ", correctos)
    print("Totales: ", len(resultados_ocr))
    print(f"Porcentaje de acierto: {100*correctos/len(resultados_ocr)}%")
    print("Tiempo de inferencia: ", sum([x[1] for x in resultados_ocr]))
    print("Tiempo de inferencia promedio: ", sum(
        [x[1] for x in resultados_ocr])/len(resultados_ocr))
    print("Tiempo de match: ", sum(tiempo_match))
    print("Tiempo de match promedio: ", sum(tiempo_match)/len(tiempo_match))
    print("Tiempo total: ", sum([x[1]
          for x in resultados_ocr])+sum(tiempo_match))
    print("Tiempo total promedio: ", (sum(
        [x[1] for x in resultados_ocr])+sum(tiempo_match))/len(resultados_ocr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B/32',
                        help='model name or path to model')
    parser.add_argument('--images', type=str, default='img/gondolas/',
                        help='input image to perform inference on')
    parser.add_argument('--labels', type=str, default='["elvive", "a dog", "a orange cat"]',
                        help='labels to compare image')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='threshold for ocr')
    parser.add_argument('--dataset', type=str, default='data/text_products.csv',
                        help='dataset to compare ocr')
    parser.add_argument('--image_dir', type=str, default='bounding_boxes/',
                        help='path to images')
    args = parser.parse_args()
    images = f'{args.images}M10-OAI-F.OAI FARMA/'
    imgs = glob.glob(images + '**/*.webp', recursive=True)
    # Clip
    # read labels will be a df with sku, description to clip:
    df = pd.read_csv('data/clip_medicamentos.csv')
    dataset_clip = 'data/clip_medicamentos.csv'
    dataset_ocr = 'data/ocr_medicamentos.csv'
    df_ocr = pd.read_csv(dataset_ocr)
    # clip_recognition(dataset_clip, args.images, imgs, args.model)
    tiemop_carga = t.time()
    model, preprocess, device = clip.load_model(args.model)
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo CLIP: ", tiempo_carga_stop-tiemop_carga)
    cantidad_imagenes = len(imgs)
    in_filter = 0
    correctos = 0
    tiemop_carga = t.time()
    engine = ocr.ocr_load_model()
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo OCR: ", tiempo_carga_stop-tiemop_carga)
    # #CLIP como filtro + OCR
    # for img in imgs:
    #     results = clip_filter(
    #         dataset_clip, (model, preprocess, device), img, 5)
    #     sku_real = int(img.split('/')[-2])
    #     # create a subdataset ocr with the results:

    #     subdataset_ocr = df_ocr[df_ocr['SKU'].isin(results)]
    #     if not subdataset_ocr.empty:
    #         predict = ocr_single_recognition(img, subdataset_ocr, engine)
    #     else:
    #         predict = ocr_single_recognition(img, df_ocr, engine)
    #         print("filtro no encontro resultados, se usa el dataset completo")
    #     print(predict)
    #     if int(predict) == sku_real:
    #         correctos += 1
    #     if sku_real in results:
    #         in_filter += 1
    # print(
    #     f"Porcentaje de imagenes en el filtro: {100*in_filter/cantidad_imagenes}%")
    # print(
    #     f"Porcentaje de imagenes correctas: {100*correctos/cantidad_imagenes}%")

    # # OCR como filtro + CLIP
    # correctos = 0
    # for img in imgs:
    #     sku_real = int(img.split('/')[-2])
    #     filtro_ocr = ocr_filter(img, df_ocr, engine, 5)
    #     print(sku_real, filtro_ocr)
    #     if sku_real in filtro_ocr:
    #         in_filter += 1
    #     subdf_clip = df[df['sku'].isin(filtro_ocr)]
    #     if not subdf_clip.empty:
    #         predict = clip_single_recognition(subdf_clip,
    #                                           img, (model, preprocess, device))
    #     else:
    #         predict = clip_single_recognition(df,
    #                                           img, (model, preprocess, device))
    #         print("filtro no encontro resultados, se usa el dataset completo")
    #     print(predict)
    #     if int(predict) == sku_real:
    #         correctos += 1

    # print(
    #     f"Porcentaje de imagenes en el filtro: {100*in_filter/cantidad_imagenes}%")
    # print(
    #     f"Porcentaje de imagenes correctas: {100*correctos/cantidad_imagenes}%")
    ocr_recognition(args.images, imgs, df_ocr)


if __name__ == '__main__':
    main()
    # diccionario = {
    #     123456: [123456, 654321, 123456, 123456],
    #     654321: [123456, 654321, 123456, 654321],
    #     321654: [123456, 321654, 654321, 321654]}
    # matriz = matriz_confusion(diccionario)
    # matriz_comprimida = matriz_confusion_comprimida(diccionario)
    # generar_curvas_roc(matriz_comprimida)
