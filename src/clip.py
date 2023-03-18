import torch
import src.CLIP.clip as clip
from PIL import Image
import time


def load_model(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model, device=device)
    return model, preprocess, device

def inference(model,preprocess,device,img,labels):
    start = time.time()
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    stop = time.time()  
    print("Tiempo de clip: ",stop-start)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)

    # Print the result
    print("\nTop predictions of " + ":\n")
    count=1

    for value, index in zip(values, indices):    
        
        print( str(count)+" lugar: "+f"{labels[index]:>16s}: {100 * value.item():.2f}%")
        count+=1


if __name__ == '__main__':
    tiemop_carga = time.time()
    model, preprocess, device = load_model("ViT-B/32")
    tiempo_carga_stop = time.time()
    print("Tiempo de carga del modelo: ",tiempo_carga_stop-tiemop_carga)
    inference(model,preprocess,device,"Cat03.jpg",["a black cat", "a dog", "a orange cat"])
    inference(model,preprocess,device,"Cat03.jpg",["a red cat", "a dog", "a orange cat"])

