import torch
import CLIP.clip as clip
from PIL import Image
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
carga_modelo = time.time()
model, preprocess = clip.load("ViT-B/32", device=device)
carga_modelo_stop = time.time()
print("Tiempo de carga del modelo: ",carga_modelo_stop-carga_modelo)
start = time.time()
image = preprocess(Image.open("Cat03.jpg")).unsqueeze(0).to(device)
label=["a diagram", "a dog", "a cat"]

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

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
    
    print( str(count)+" lugar: "+f"{label[index]:>16s}: {100 * value.item():.2f}%")
    count+=1