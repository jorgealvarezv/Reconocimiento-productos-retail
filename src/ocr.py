import src.PaddleOCR.paddleocr as ocr
from difflib import SequenceMatcher
import pandas as pd
import ast
import time as t


args = ocr.parse_args()

def ocr_inference(args):
    results,tiempo,imgs= ocr.test(args)
    tiempo_init=tiempo[0]
    #eliminar el tiempo de inicialización
    tiempo.pop(0)
    results.pop(0)
    imgs.pop(0)
    final=[]
    for result in results:
        palabras=[]
        for i in result:
            palabras+=i.replace("-"," ").replace("/"," ").replace("."," ").replace(","," ").split(' ')
        
        final.append(palabras)
    return final,tiempo,tiempo_init,imgs

def ocr_match(list_words_bd,ocr_words,threshold):
    
    text = list_words_bd
    text_list = ocr_words
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
                    if(index_bd == 0 or index_bd == 1):
                        score += 4
                        break
                    elif(index_bd == 2 or index_bd == 3):
                        score += 2
                        break
                    else:
                        score += 1
                        break
            else:
                
                
                matcher = SequenceMatcher(None,palabra_bd,palabra_ocr)
                match = matcher.find_longest_match(0,
                                        largo_palabra_bd,
                                        0,
                                        largo_palabra_ocr)
                cadena_mas_larga = match.size
                if cadena_mas_larga/largo_palabra_bd > threshold:
                    
                    if(index_bd == 0 or index_bd == 1):
                        score += 4
                        break
                    elif(index_bd == 2 or index_bd == 3):
                        score += 2
                        break
                    else:
                        score += 1
                        break
            index_ocr += 1
        index_bd += 1
    
    
    return score            

if __name__ == '__main__':
    results,tiempo,init,imgs=ocr_inference(args)
    
    print("\nTiempo de inicialización: ",init)
    dataset='data/text_products.csv'
    dataframe = pd.read_csv(dataset)
    
    
    for result,time,img in zip(results,tiempo,imgs):
        print("----"*35)
        print(f"Se encontrarón {len(result)} palabras en la imagen {img} en un tiempo de {time} segundos")
        print(result)
        print("----"*35)
        start_match = t.time()
        skus=[]
        for (sku, text) in zip(dataframe['SKU'], dataframe['Text']):
                sku = str(sku).zfill(6)
                text = ast.literal_eval(text)
                score=ocr_match(text,result,0.8)
                if(score!=0):
                    skus.append((sku,score))
        end=t.time()
        print("Tiempo de match: ",end-start_match)
        #ordena skus por score de mayor a menor e imprime el primero
        skus.sort(key=lambda tup: tup[1],reverse=True)
        print(f"La imagen {img} coincide con el SKU ",skus[0][0])
        print(f"Tiempo total fue {time+end-start_match}")
        


