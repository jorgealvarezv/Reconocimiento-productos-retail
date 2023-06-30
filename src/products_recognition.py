
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

MAX_DISTANCE = 2
MIN_SCORE = 4
FILTRO_OCR = 10
FILTRO_CLIP = 10
TRESHOLD_OCR = 0.8


def matriz_confusion(diccionario, name="matriz_confusion_clip"):
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
    # ruta_archivo = 'matriz_confusion_clip.csv'
    # matriz.to_csv(ruta_archivo, index=False)
    # ruta_archivo = 'matriz_confusion_clip.xlsx'
    # matriz.to_excel(ruta_archivo, index=False)

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
    title = "Matriz Confusion"+name.replace("_", " ")
    ax.set_title(title)
    ax.title.set_size(64)

    # Mostrar el mapa de calor
    plt.savefig(f'matriz_confusion_{name}.png', dpi=300)
    return matriz


def matriz_confusion_comprimida(diccionario, name="matriz_confusion_clip"):
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
    title = "Matriz Confusion" + name.replace("_", " ")
    ax.set_title(title)
    plt.tight_layout()

    # Guardar el gráfico en un archivo

    plt.savefig(f'matriz_confusion_{name}.png', dpi=300)

    return matriz_


def generar_curvas_roc(matriz_confusion, name="curva_roc_clip"):
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
    title = "Curva ROC"+name.replace("_", " ")
    plt.title(title)
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend()
    plt.savefig(f'curva_roc_{name}.png', dpi=300)


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


def ocr_filter(img, dataframe, model, results=FILTRO_OCR):
    """Uso de ocr que recibe todas las imagenes y devuelve la cantidad dada de resultados"""

    words, _ = ocr.ocr_inference(model, img, TRESHOLD_OCR)
    # print(words)
    sku = int(img.split('/')[-2])
    better_score = 0
    skus = []
    for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
        sku = str(sku).zfill(6)
        text = ast.literal_eval(text)
        score = ocr.ocr_match(text, words, TRESHOLD_OCR)
        # print(sku, score)
        if score > better_score:
            better_score = score
        # if score >= better_score - MAX_DISTANCE and score >= MIN_SCORE:
        skus.append((sku, score))
        # print(sku, score)
    skus = [(sku, score) for (sku, score) in skus if score >=
            better_score - MAX_DISTANCE and score >= MIN_SCORE]
    # print(skus)
    skus.sort(key=lambda tup: tup[1], reverse=True)
    # print(skus)
    if (len(skus) == 0):
        return []
    filtro = [int(sku) for sku, _ in skus]
    return filtro[:results]


def ocr_single_recognition(img, dataframe, model):
    """with a img and df ocr do the match"""
    words, _ = ocr.ocr_inference(model, img, TRESHOLD_OCR)
    # print(words)
    sku = int(img.split('/')[-2])

    skus = []
    better_score = 0
    for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
        sku = str(sku).zfill(6)
        text = ast.literal_eval(text)

        score = ocr.ocr_match(text, words, TRESHOLD_OCR)
        if score > better_score:
            better_score = score
        # if score >= better_score - MAX_DISTANCE and score >= MIN_SCORE:
        skus.append((sku, score))
    skus = [(sku, score) for (sku, score) in skus if score >=
            better_score - MAX_DISTANCE and score >= MIN_SCORE]
    skus.sort(key=lambda tup: tup[1], reverse=True)
    # print(skus)
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
    matriz_ocr = matriz_confusion(dict_resultados_ocr, "solo_ocr")
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


def only_clip(imgs, csv_clip, model_clip, dict):
    model, preprocess, device = model_clip
    correctos = 0
    cantidad_imagenes = len(imgs)
    for img in imgs:
        results = clip_filter(
            csv_clip, (model, preprocess, device), img, 1)
        sku_real = int(img.split('/')[-2])
        predict = results[0]
        if (predict == sku_real):
            correctos += 1
        dict[sku_real].append(int(predict))
    print(
        f"Porcentaje de imagenes correctas: {100*correctos/cantidad_imagenes}%")
    return dict
    # return dict, correctos/cantidad_imagenes


def only_ocr(imgs, csv_ocr, model_ocr, dict):
    """recognition with ocr filter and clip"""
    df_ocr = pd.read_csv(csv_ocr)
    correctos = 0
    cantidad_imagenes = len(imgs)
    for img in imgs:
        results = ocr_filter(img, df_ocr, model_ocr, results=1)
        sku_real = int(img.split('/')[-2])
        if results == []:
            predict = 0
        else:
            predict = results[0]
        if predict == sku_real:
            correctos += 1
        dict[sku_real].append(int(predict))
    print(
        f"Porcentaje de imagenes correctas: {100*correctos/cantidad_imagenes}%")
    return dict


def recognition_1(imgs, csv_clip, csv_ocr, model_clip, model_ocr, dict):
    """recognition with clip filter and ocr"""
    df_ocr = pd.read_csv(csv_ocr)
    model, preprocess, device = model_clip
    correctos = 0
    in_filter = 0
    cantidad_imagenes = len(imgs)
    for img in imgs:
        results = clip_filter(
            csv_clip, (model, preprocess, device), img, FILTRO_CLIP)
        sku_real = int(img.split('/')[-2])
        # create a subdataset ocr with the results:

        subdataset_ocr = df_ocr[df_ocr['SKU'].isin(results)]
        if not subdataset_ocr.empty:
            predict = ocr_single_recognition(img, subdataset_ocr, model_ocr)
        else:
            predict = ocr_single_recognition(img, df_ocr, model_ocr)
            # print("filtro no encontro resultados, se usa el dataset completo")
        # print(predict)
        if int(predict) == sku_real:
            correctos += 1
        if sku_real in results:
            in_filter += 1
        dict[sku_real].append(int(predict))
    print(
        f"Porcentaje de imagenes en el filtro: {100*in_filter/cantidad_imagenes}%")
    print(
        f"Porcentaje de imagenes correctas: {100*correctos/cantidad_imagenes}%")
    return dict


def recognition_2(imgs, csv_clip, csv_ocr, model_clip, model_ocr, dict):
    """recognition with ocr filter and clip"""
    df_ocr = pd.read_csv(csv_ocr)
    model, preprocess, device = model_clip
    correctos = 0
    in_filter = 0
    cantidad_imagenes = len(imgs)
    for img in imgs:
        results = ocr_filter(img, df_ocr, model_ocr, results=FILTRO_OCR)
        # print(results)
        sku_real = int(img.split('/')[-2])
        # create a subdataset ocr with the results:
        subdataset_clip = pd.read_csv(csv_clip)
        # print(results)
        subdataset_clip = subdataset_clip[subdataset_clip['sku'].isin(results)]
        # print(subdataset_clip)
        # subdataset_clip = subdataset_clip[subdataset_clip['sku'].isin(results)]
        if not subdataset_clip.empty:
            predict = clip_single_recognition(subdataset_clip, img, model_clip)
        else:
            predict = clip_single_recognition(pd.read_csv(csv_clip),
                                              img, model_clip)
            # print("filtro no encontro resultados, se usa el dataset completo")
        # print(predict)
        if int(predict) == sku_real:
            correctos += 1
        if sku_real in results:
            in_filter += 1
        dict[sku_real].append(int(predict))
    print(
        f"Porcentaje de imagenes en el filtro: {100*in_filter/cantidad_imagenes}%")
    print(
        f"Porcentaje de imagenes correctas: {100*correctos/cantidad_imagenes}%")
    return dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RN50',
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
    # images = f'{args.images}2P-A-BAJO.ALIMENTO/'
    # images = f'{args.images}M10-OAI-F.OAI FARMA/'
    images = f'{args.images}1M5-RF.BIENESTAR SEXUAL/'
    imgs = glob.glob(images + '**/*.webp', recursive=True)
    cantidad_imagenes = len(imgs)
    # Clip
    dataset_clip = 'data/clip_BIENESTAR.csv'
    dataset_ocr = 'data/ocr_BIENESTAR.csv'
    # dataset_clip = 'data/clip_ALIMENTO.csv'
    # dataset_ocr = 'data/ocr_ALIMENTO.csv'
    # dataset_clip = 'data/clip_medicamentos.csv'
    # dataset_ocr = 'data/ocr_medicamentos.csv'
    df_ocr = pd.read_csv(dataset_ocr)
    # clip_recognition(dataset_clip, args.images, imgs, args.model)
    tiemop_carga = t.time()
    model, preprocess, device = clip.load_model(args.model)
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo CLIP: ",
          tiempo_carga_stop-tiemop_carga)
    tiemop_carga = t.time()
    engine = ocr.ocr_load_model()
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo OCR: ", tiempo_carga_stop-tiemop_carga)
    # df_models = pd.DataFrame(columns=[
    #                          "Modelo", "Tiempo de carga", "Tiempo por imagen", "Porcentaje de imagenes correctas"])
    # for model_name in os.listdir("models/"):
    #     dict_resultados = {
    #         int(img.replace(images, "")[:6]): [] for img in imgs}
    #     tiemop_carga = t.time()
    #     model_name = model_name.replace(".pkl", "").replace("model_", "")
    #     if ("preprocess" in model_name):
    #         continue
    #     print("Modelo: ", model_name)
    #     model, preprocess, device = clip.load_model(model_name)
    #     tiempo_carga_stop = t.time()
    #     print("Tiempo de carga del modelo CLIP: ",
    #           tiempo_carga_stop-tiemop_carga)
    #     start_3 = t.time()
    #     results_3, correctos = only_clip(imgs, dataset_clip, (
    #         model, preprocess, device), dict_resultados)
    #     end_3 = t.time()
    #     print(
    #         f"Tiempo de ejecucion de only_clip total {end_3-start_3} Por producto: {(end_3-start_3)/cantidad_imagenes}")
    #     #     start_1 = t.time()

    #     #     results_1 = recognition_1(imgs, dataset_clip, dataset_ocr, (
    #     #         model, preprocess, device), engine, dict_resultados)
    #     #     end_1 = t.time()
    #     #     print(
    #     #         f"Tiempo de ejecucion de clip_filter+ocr total {end_1-start_1} Por producto: {(end_1-start_1)/cantidad_imagenes}")
    #     #     start_2 = t.time()
    #     #     results_2 = recognition_2(imgs, dataset_clip, dataset_ocr, (
    #     #         model, preprocess, device), engine, dict_resultados)
    #     #     end_2 = t.time()
    #     #     print(
    #     #         f"Tiempo de ejecucion de ocr_filter+clip total {end_2-start_2} Por producto: {(end_2-start_2)/cantidad_imagenes}")

    #     df_models = df_models.append({"Modelo": model_name, "Tiempo de carga": tiempo_carga_stop-tiemop_carga, "Tiempo por imagen": (
    #         end_3-start_3)/cantidad_imagenes, "Porcentaje de imagenes correctas": correctos}, ignore_index=True)
    # print(df_models)

    # ocr_recognition(args.images, imgs, df_ocr)

    dict_resultados = {
        int(img.replace(images, "")[:6]): [] for img in imgs}
    start_1 = t.time()
    results_1 = recognition_1(imgs, dataset_clip, dataset_ocr, (
        model, preprocess, device), engine, dict_resultados)
    end_1 = t.time()
    print(
        f"Tiempo de ejecucion de clip_filter+ocr total {end_1-start_1} Por producto: {(end_1-start_1)/cantidad_imagenes}")

    matriz_confusion(results_1, "clip_filter+ocr")
    results_1_ = matriz_confusion_comprimida(
        results_1, "clip_filter+ocr_comprimida")
    generar_curvas_roc(results_1_, "clip_filter+ocr")

    dict_resultados = {
        int(img.replace(images, "")[:6]): [] for img in imgs}

    start_2 = t.time()
    results_2 = recognition_2(imgs, dataset_clip, dataset_ocr, (
        model, preprocess, device), engine, dict_resultados)
    end_2 = t.time()
    print(
        f"Tiempo de ejecucion de ocr_filter+clip total {end_2-start_2} Por producto: {(end_2-start_2)/cantidad_imagenes}")
    matriz_confusion(results_2, "ocr_filter+clip")
    results_2_ = matriz_confusion_comprimida(
        results_2, "ocr_filter+clip_comprimida")
    generar_curvas_roc(results_2_, "ocr_filter+clip")

    dict_resultados = {
        int(img.replace(images, "")[:6]): [] for img in imgs}

    start_3 = t.time()
    results_3 = only_clip(imgs, dataset_clip, (
        model, preprocess, device), dict_resultados)
    end_3 = t.time()
    print(
        f"Tiempo de ejecucion de only_clip total {end_3-start_3} Por producto: {(end_3-start_3)/cantidad_imagenes}")
    matriz_confusion(results_3, "only_clip")
    results_3_ = matriz_confusion_comprimida(results_3, "only_clip_comprimida")
    generar_curvas_roc(results_3_, "only_clip")

    dict_resultados = {
        int(img.replace(images, "")[:6]): [] for img in imgs}

    start_4 = t.time()
    results_4 = only_ocr(imgs, dataset_ocr, engine, dict_resultados)
    end_4 = t.time()
    print(
        f"Tiempo de ejecucion de only_ocr total {end_4-start_4} Por producto: {(end_4-start_4)/cantidad_imagenes}")
    matriz_confusion(results_4, "only_ocr")
    results_4_ = matriz_confusion_comprimida(results_4, "only_ocr_comprimida")
    generar_curvas_roc(results_4_, "only_ocr")


if __name__ == '__main__':
    main()
    # dataset_clip = 'data/clip_ALIMENTO.csv'
    # dataset_ocr = 'data/ocr_ALIMENTO.csv'
    # SKU = '267186'
    # images = f'img/gondolas/2P-A-BAJO.ALIMENTO/{SKU}/'
    # images_result = 'img/gondolas/2P-A-BAJO.ALIMENTO/'
    # imgs = glob.glob(images + '**/*.webp', recursive=True)

    # df_ocr = pd.read_csv(dataset_ocr)
    # df_clip = pd.read_csv(dataset_clip)
    # tiempo_carga = t.time()
    # model, preprocess, device = clip.load_model('RN50')
    # tiempo_carga_stop = t.time()
    # print("Tiempo de carga del modelo CLIP: ", tiempo_carga_stop-tiempo_carga)

    # tiempo_carga = t.time()
    # engine = ocr.ocr_load_model()
    # tiempo_carga_stop = t.time()
    # print("Tiempo de carga del modelo OCR: ", tiempo_carga_stop-tiempo_carga)

    # for img in imgs:
    #     predict_ocr = ocr_single_recognition(img, df_ocr, engine)
    #     predict_clip = clip_single_recognition(
    #         df_clip, img, (model, preprocess, device))
    #     print(f"Imagen: {img} OCR: {predict_ocr} CLIP: {predict_clip}")
    # dict_resultados = {
    #     int(img.replace(images_result, "")[:6]): [] for img in imgs}
    # predict_mix_1 = recognition_1(imgs, dataset_clip, dataset_ocr, (
    #     model, preprocess, device), engine, dict_resultados)
    # print(f'CLIP+OCR: {predict_mix_1}')
    # dict_resultados = {
    #     int(img.replace(images_result, "")[:6]): [] for img in imgs}
    # predict_mix_2 = recognition_2(imgs, dataset_clip, dataset_ocr, (
    #     model, preprocess, device), engine, dict_resultados)
    # print(f'OCR+CLIP: {predict_mix_2}')
