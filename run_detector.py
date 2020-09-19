from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_model_instance_segmentation(num_classes, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return model


@st.cache(show_spinner=False)
def get_mask_model():
    model = get_model_instance_segmentation(3, pretrained=False)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache(show_spinner=False)
def get_face_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model


def predict(img_data):
    with st.spinner('Loading AI model...'):
        mask_model = get_mask_model()
        face_model = get_face_model()
    with st.spinner('Running AI model...'):
        img = ToTensor()(Image.open(img_data)).unsqueeze(0)
        mask_pred = mask_model(img)
        face_pred = face_model(img)
        st.write(face_pred[0]['labels'].shape)
        st.write(len(face_pred[0]['boxes']))
    with st.spinner('Processing Results...'):
        img = img[0].cpu().data
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0))
        plot_annotated_image(fig, ax, mask_pred[0], color='r')
        plot_annotated_image(fig, ax, face_pred[0], color='b')
        ax.axis('off')
        st.pyplot()

    st.success('Your image has been processed!')


def plot_annotated_image(fig, ax, annotation, color='r'):
    for label, box in zip(annotation["labels"], annotation["boxes"]):
        if label != 1:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)
