import src.clip as clip
import src.ocr as ocr

import time as t
import argparse
import pandas as pd
from PIL import Image
import ast

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='ViT-B/32',
    #                     help='model name or path to model')
    # parser.add_argument('--image', type=str, default='Cat03.jpg',
    #                     help='input image to perform inference on')
    # parser.add_argument('--labels', type=str, default='["a black cat", "a dog", "a orange cat"]',
    #                     help='labels to compare image')
    # parser.add_argument('--threshold', type=float, default=0.8,
    #                     help='threshold for ocr')
    # parser.add_argument('--dataset', type=str, default='data/text_products.csv',
    #                     help='dataset to compare ocr')
    parser.add_argument('--image_dir', type=str, default='bounding_boxes/',
                        help='path to images')
    args = parser.parse_args()
    
    # Clip
    # tiemop_carga = t.time()
    # model, preprocess, device = clip.load_model(args.model)
    # tiempo_carga_stop = t.time()
    # print("Tiempo de carga del modelo: ",tiempo_carga_stop-tiemop_carga)
    # img=Image.open(args.image)
    # clip.inference(model,preprocess,device,img,ast.literal_eval(args.labels))
    
    # OCR
    results,tiempo,init,imgs=ocr.ocr_inference(args)
    print("\nTiempo de inicialización: ",init)
    dataframe = pd.read_csv('data/text_products.csv')
    for result,time,img in zip(results,tiempo,imgs):
        print("\nTiempo de ocr: ",time)
        print("\nScore de ocr: ",result)
        print("\nImagen: ",img)
        print("\n")
        print(dataframe.loc[dataframe['image'] == img])
        print("\n")
        print("----------------------------------------------------------")
        print("\n")

if __name__ == '__main__':
    main()
