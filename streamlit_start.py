import io
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import gdown

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

def load_model():
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    model_1 = smp.Unet('resnet34', in_channels=1,
                       encoder_weights='imagenet',
                       classes=1, activation=None,
                       encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.eval()
    return model


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(img):
    image = img.resize((224, 224))
    x = np.array(image)
    x = np.array(x / 255, dtype='float32')
    x = transform(x)
    x = torch.reshape(x, (1, 3, 224, 224))
    return x

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def print_predicitions(output):
    results = utils.pick_n_best(predictions=output, n=3)
    for cl in results:
        st.write(cl[1], cl[2])

url = 'https://drive.google.com/uc?id=1jYRFlxiGq6XTRWd0TR0NVCdfGKh7Sw1y'
output = 'model_resnet.pth'
gdown.download(url, output, quiet=False)
        
model = load_model()
model_1 = torch.load('model_resnet.pth')

st.title('Новая улучшенная классификация изображений в облаке streamlit')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    with torch.no_grad():
        preds = torch.nn.functional.softmax(model(x), dim=1)
    st.write('Результаты распознавания')
    print_predicitions(preds)
    a = x.detach().to('cpu').numpy().squeeze()
    b = np.rollaxis(a, 0, 3)
    st.image(b, clamp=True)
