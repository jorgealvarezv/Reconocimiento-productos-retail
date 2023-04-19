from PaddleOCR import paddleocr
from difflib import SequenceMatcher
import pandas as pd
import ast
import time as t
import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
ocr_args = paddleocr.parse_args(mMain=True)


def ocr_load_model():
    engine = paddleocr.PaddleOCR(**(ocr_args.__dict__))
    return engine


def ocr_inference(engine, img, treshold):
    start = t.time()
    result = engine.ocr(img,
                        det=ocr_args.det,
                        rec=ocr_args.rec,
                        cls=ocr_args.use_angle_cls)
    stop = t.time()
    textos = [line[1][0].lower().replace("-",
              " ").replace("/"," ").replace(".",
              " ").replace(",", " ").replace("'", "")
              for sublist in result for line in sublist if (line[1][1] > treshold)]
    textos_final = [palabra
                    for palabras in textos for palabra in palabras.split(' ')]

    # if result is not None:
    #     for idx in range(len(result)):
    #         res = result[idx]
    #         resultados=[line[1][0].lower().replace("-",
    #         " ").replace("/"," ").replace(".",
    #         " ").replace(",", " ").split(' ') for line in res if(line[1][1]>0.5)]

    # palabras = []
    # for i in resultados:
    #     palabras += i.replace("-",
    #     " ").replace("/"," ").replace(".",
    #     " ").replace(",", " ").split(' ')
    #     final.append(palabras)
    return textos_final, stop-start


def ocr_match(list_words_bd, ocr_words, threshold):

    n_words_bd = len(list_words_bd)
    n_ocr_words = len(ocr_words)
    score = 0
    index_bd = 0

    while index_bd < n_words_bd:

        index_ocr = 0

        while index_ocr < n_ocr_words:
            palabra_bd = list_words_bd[index_bd]
            largo_palabra_bd = len(palabra_bd)
            palabra_ocr = ocr_words[index_ocr]
            largo_palabra_ocr = len(palabra_ocr)
            if largo_palabra_bd == largo_palabra_ocr:
                if palabra_bd == palabra_ocr:
                    if (index_bd == 0 or index_bd == 1):
                        score += 4
                        break
                    elif (index_bd == 2 or index_bd == 3):
                        score += 2
                        break
                    else:
                        score += 1
                        break
            else:

                matcher = SequenceMatcher(None, palabra_bd, palabra_ocr)
                match = matcher.find_longest_match(0,
                                                   largo_palabra_bd,
                                                   0,
                                                   largo_palabra_ocr)
                cadena_mas_larga = match.size
                if cadena_mas_larga/largo_palabra_bd > threshold:

                    if (index_bd == 0 or index_bd == 1):
                        score += 4
                        break
                    elif (index_bd == 2 or index_bd == 3):
                        score += 2
                        break
                    else:
                        score += 1
                        break
            index_ocr += 1
        index_bd += 1

    return score
def ocr_match_2(list_words_bd, ocr_words, threshold):
    score = 0

    for index_bd, palabra_bd in enumerate(list_words_bd):
        largo_palabra_bd = len(palabra_bd)
        match_found = False

        for index_ocr, palabra_ocr in enumerate(ocr_words):
            largo_palabra_ocr = len(palabra_ocr)

            if largo_palabra_bd == largo_palabra_ocr:
                if palabra_bd == palabra_ocr:
                    match_found = True
                    break
            else:
                matcher = SequenceMatcher(None, palabra_bd, palabra_ocr)
                match = matcher.find_longest_match(0, largo_palabra_bd, 0, largo_palabra_ocr)
                cadena_mas_larga = match.size

                if cadena_mas_larga/largo_palabra_bd > threshold:
                    match_found = True
                    break

        if match_found:
            if index_bd in [0, 1]:
                score += 4
            elif index_bd in [2, 3]:
                score += 2
            else:
                score += 1

    return score

if __name__ == '__main__':
    results, tiempo, init, imgs = ocr_inference(args)

    print("\nTiempo de inicialización: ", init)
    dataset = 'data/text_products.csv'
    dataframe = pd.read_csv(dataset)

    for result, time, img in zip(results, tiempo, imgs):
        print("----"*35)
        print(
            f"Se encontrarón {len(result)} palabras en la imagen {img} en un tiempo de {time} segundos")
        print(result)
        print("----"*35)
        start_match = t.time()
        skus = []
        for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
            sku = str(sku).zfill(6)
            text = ast.literal_eval(text)
            score = ocr_match(text, result, 0.8)
            if (score != 0):
                skus.append((sku, score))
        end = t.time()
        print("Tiempo de match: ", end-start_match)
        # ordena skus por score de mayor a menor e imprime el primero
        skus.sort(key=lambda tup: tup[1], reverse=True)
        print(f"La imagen {img} coincide con el SKU ", skus[0][0])
        print(f"Tiempo total fue {time+end-start_match}")
