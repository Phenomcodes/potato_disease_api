from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import os
import gdown
from  cnn import model

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH =os.path.join(BASE_DIR, 'model.pth')

if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/file/d/1pBGYevGbizr--Wl3eg5cTjKD5diye1BF/view?usp=drive_link",
        MODEL_PATH,
        quiet=False
    )

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()
classes = ["Healthy", "Early_blight", "Late_blight"]

valid_transform = transforms.Compose([
    transforms.CenterCrop([128, 128]),
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])

def read_file_as_image(data):
    image = Image.open(io.BytesIO(data))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = valid_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = MODEL(image)
        _, predicted = torch.max(output, 1)
        probs = torch.softmax(output, 1)
        confidence = probs[0][predicted[0]].item()
    return {"predicted": classes[predicted[0]], "confidence": int(100*confidence)}

    #print("Predicted class:", classes[predicted.item()])
    #print("Confidence level:", int(100 * confidence))
