"""Uso de CLIP para memoria de reconocimiento de productos"""
import torch
import time
import pickle
from PIL import Image
from CLIP import clip


def save_model(model_name):
    """Save model to using pickle"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load(model_name, device=device)
    model_name=model_name.replace("/", "-")
    with open(f'models/model_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'models/preprocess_{model_name}.pkl', 'wb') as f:
        pickle.dump(preprocess, f)

def load_model(model_name):
    """Load model from pickle"""
    model_name=model_name.replace("/", "-")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(f'models/model_{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/preprocess_{model_name}.pkl', 'rb') as f:
        preprocess = pickle.load(f)
    return model, preprocess, device

    

def inference(model,preprocess,device,img,df):
    """Inference with CLIP"""
    tiempo_start = time.time()
    img_=Image.open(img)
    image = preprocess(img_).unsqueeze(0).to(device)

    skus = df['sku'].tolist()
    labels = df['description'].tolist()
    
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    value, indice = similarity[0].topk(1)
    tiempo_stop = time.time()
    tiempo_inferencia=tiempo_stop-tiempo_start
    return skus[indice],100 * value.item(),tiempo_inferencia 


if __name__ == '__main__':
    tiemop_carga = time.time()
    model, preprocess, device = load_model("ViT-B/32")
    tiempo_carga_stop = time.time()
    print("Tiempo de carga del modelo: ",tiempo_carga_stop-tiemop_carga)
    inference(model,preprocess,device,"Cat03.jpg",["a black cat", "a dog", "a orange cat"])
    inference(model,preprocess,device,"Cat03.jpg",["a red cat", "a dog", "a orange cat"])

