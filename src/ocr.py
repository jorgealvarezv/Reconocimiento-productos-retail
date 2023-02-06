import PaddleOCR.paddleocr as ocr
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
            if len(i.split(' '))>1:
                palabras+=i.replace("-"," ").replace(","," ").split(' ')
            else:
                palabras.append(i)
        final.append(palabras)
    return final,tiempo,tiempo_init,imgs

if __name__ == '__main__':
    results,tiempo,init,imgs=ocr_inference(args)
    
    print("\nTiempo de inicialización: ",init)

    for result,time,img in zip(results,tiempo,imgs):
        print("----"*35)
        print(f"Se encontrarón {len(result)} palabras en la imagen {img} en un tiempo de {time} segundos")
        print(result)

