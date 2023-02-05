import PaddleOCR.paddleocr as ocr
args = ocr.parse_args()

def ocr_inference(args):
    results,tiempo= ocr.test(args)
    return results,tiempo

if __name__ == '__main__':
    results,tiempo=ocr_inference(args)
    print(results)
