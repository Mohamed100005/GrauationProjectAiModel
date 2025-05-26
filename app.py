import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class_names = [
    'Dermatitis',
    'Fungal_infections',
    'Healthy',
    'Hypersensitivity',
    'demodicosis',
    'ringworm'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
@st.cache_resource
def load_model():
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

 
st.title("🐶 Dog Skin Disease Classifier")
st.write("قم برفع صورة لجلد الكلب لتحليلها والتعرف على المرض.")

file = st.file_uploader("📤 اختر صورة", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="📷 الصورة المرفوعة", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()

    predicted_class = class_names[predicted_idx]
    confidence = round(probabilities[predicted_idx].item() * 100, 2)

    st.success(f"✅ التشخيص: **{predicted_class}** بنسبة ثقة {confidence}%")

    st.subheader("📊 نسب باقي التصنيفات:")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {round(probabilities[i].item() * 100, 2)}%")
        st.progress(probabilities[i].item())
