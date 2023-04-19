
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
    
    #Clip
    #read labels will be a df with sku, description to clip:
    df = pd.read_csv('data/clip_medicamentos.csv')
    labels = df['description'].tolist()
    
    images = f'{args.images}M10-OAI-F.OAI FARMA/'
    tiemop_carga = t.time()
    #clip.save_model(args.model)
    model, preprocess, device = clip.load_model(args.model)
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo CLIP: ",tiempo_carga_stop-tiemop_carga)
    images_ocr=os.listdir(images)
    imgs = glob.glob(images + '**/*.webp', recursive=True)

    #print(imgs)
    resultados_clip=[ (clip.inference(model,preprocess,device,img,df),int(img.replace(images,"")[:6])) for img in imgs ]
    correctos=0
    totales=len(resultados_clip)
    tiempo_total=0
    for result,sku in resultados_clip:
        tiempo_total+=result[2]
        if(result[0]==sku):
            correctos+=1
    print("Correctos: ",correctos)
    print("Totales: ",totales)
    print("Porcentaje de acierto: ",100*correctos/totales)
    print("Tiempo de inferencia: ",tiempo_total)
    print("Tiempo de inferencia promedio: ",tiempo_total/totales)
        
    
    # # OCR
    tiemop_carga = t.time()
    engine=ocr.ocr_load_model()
    tiempo_carga_stop = t.time()
    print("Tiempo de carga del modelo OCR: ",tiempo_carga_stop-tiemop_carga)
    resultados_ocr=[]
    for img in imgs:
        words,t_inference=ocr.ocr_inference(engine,img,0)
        sku=int(img.replace(images,"")[:6])
        #print(f"Tiempo de inferencia: {t_inference}")
        resultados_ocr.append((words,t_inference,sku))
    dataset = 'data/ocr_medicamentos.csv'
    dataframe = pd.read_csv(dataset)
    tiempo_match=[]
    correctos=0
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
        if(len(skus)>0):
            sku=skus[0][0]
            if(int(sku)==int(img)):
                correctos+=1
                #print(f"Correcto: {sku} - {img}")
        end = t.time()
        #print("Tiempo de match: ", end-start_match)
        tiempo_match.append(end-start_match)

    print("Correctos: ",correctos)
    print("Totales: ",len(resultados_ocr))
    print(f"Porcentaje de acierto: {100*correctos/len(resultados_ocr)}%")
    print("Tiempo de inferencia: ",sum([x[1] for x in resultados_ocr]))
    print("Tiempo de inferencia promedio: ",sum([x[1] for x in resultados_ocr])/len(resultados_ocr))
    print("Tiempo de match: ",sum(tiempo_match))
    print("Tiempo de match promedio: ",sum(tiempo_match)/len(tiempo_match))
    print("Tiempo total: ",sum([x[1] for x in resultados_ocr])+sum(tiempo_match))
    print("Tiempo total promedio: ",(sum([x[1] for x in resultados_ocr])+sum(tiempo_match))/len(resultados_ocr))

    


if __name__ == '__main__':
    main()
    