import io
from PIL import Image
import streamlit as st
import numpy as np
import torch
from torchvision import transforms

MODEL_PATH = 'hs5-4.pth'
LABELS_PATH = 'Categories.txt'


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    all_prob, all_catid = torch.topk(probabilities, len(categories))
    for i in range(all_prob.size(0)):
        st.write(categories[all_catid[i]], all_prob[i].item())


def main():
    st.title('HS5 Computer Vision Detection')
    st.caption('This computer vision can only predict and detect 5 hate symbols which are Nazi swastika symbol, Celtic cross symbol, Sonnenrad Symbol, Schutzstaffel Symbol and Islamic State Flag symbol.')
    st.markdown('the first one to the list is the predicted outcome of the computer vision')
    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)


if __name__ == '__main__':
    main()

