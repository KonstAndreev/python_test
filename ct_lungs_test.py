import io
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import gdown

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = 'cpu'
print(f'Using {device} for inference')

def load_model():
    url = 'https://drive.google.com/uc?id=1jYRFlxiGq6XTRWd0TR0NVCdfGKh7Sw1y'
    output = 'model_resnet.pth'
    gdown.download(url, output, quiet=False)
    model = smp.Unet('resnet34', in_channels=1,
                      encoder_weights='imagenet',
                      classes=1, activation=None,
                      encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load('model_resnet.pth', map_location=torch.device('cpu')))
    return model


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_norm = transforms.Compose([
            transforms.ToTensor(),
        ])

def preprocess_image(img):
    image = img.resize((224, 224))
    x = np.array(image)
    x = np.array(x / 255, dtype='float32')
    x = transform(x)
    x = torch.reshape(x, (1, 3, 224, 224))
    return x

def load_image():
    uploaded_file = st.file_uploader(label='Загрузите файл изображения')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, channels="BGR", output_format="PNG")
        return plt.imread(io.BytesIO(image_data))
    else:
        return None

model = load_model()
st.title('Распознавание инфекции в легких')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = transform_norm(img)
    x = torch.reshape(x, (1, 1, 256, 256))
    model.eval()
    with torch.no_grad():
        Y_pred = model(x).detach().to('cpu')
    st.write('Результаты распознавания')
    a = np.array(((Y_pred.squeeze() > 0.5).int())*255)
    st.write(a[135][100])
    st.image(a, output_format="PNG", clamp=False)
