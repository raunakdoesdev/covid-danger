from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yoloface import *
import copy

st.set_option('deprecation.showPyplotGlobalUse', False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_model_instance_segmentation(num_classes, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return model


@st.cache(show_spinner=False)
def get_mask_model():
    model = get_model_instance_segmentation(3, pretrained=False)
    model.load_state_dict(torch.load('models/model.pt', map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_face_model():
    model_cfg = 'cfg/yolov3-face.cfg'
    model_weights = 'models/yolov3-wider_16000.weights'

    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def run_face_model(net, img):
    import numpy as np
    frame = np.asarray(img)[:, :, ::-1].copy()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    return faces
    # st.write(faces)


@st.cache(allow_output_mutation=True, show_spinner=False)
def run_model(img_data):
    with st.spinner('Loading AI model...'):
        mask_model = get_mask_model()
        face_model = get_face_model()

    with st.spinner('Running AI model...'):
        pil_img = Image.open(img_data)
        img = ToTensor()(pil_img).unsqueeze(0)
        face_pred = run_face_model(face_model, pil_img)
        face_pred = [face for face in face_pred if face[2] * face[3] > 300]
        mask_pred = mask_model(img)

        # filter out non-mask predictions
        mask_pred = [box for label, box in
                     zip(mask_pred[0]['labels'], mask_pred[0]['boxes']) if label == 1]
        new_mask_pred = []
        for box in mask_pred:
            xmin, ymin, xmax, ymax = box
            new_mask_pred.append((xmin.item(), ymin.item(), (xmax - xmin).item(), (ymax - ymin).item()))
        mask_pred = new_mask_pred
    return pil_img, img, mask_pred, face_pred


def predict(img_data, env):
    pil_img, img, mask_pred, face_pred = run_model(img_data)

    with st.spinner('Processing Results...'):
        img = img[0].cpu().data
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0))
        bad, good = matching(mask_pred, face_pred)

        plot_faces_annotated(fig, ax, good, color='g')
        plot_faces_annotated(fig, ax, bad, color='r')
        ax.axis('off')
        st.pyplot()
        st.markdown(f'## **{100*len(good)/(len(good)+len(bad)):.2f}%** of Individuals are Masked')


        st.markdown(f'## COVID Danger Score is **{round(10*len(bad)/(len(good)+len(bad))) + (1 if env == "Indoor" else 0)}**')
        import plotly.express as px
        fig = px.bar(x=['Mask', 'No Mask'], y=[len(good), len(bad)],
                     labels={'x': 'Mask Status', 'y': '# of Detected Faces'}, title='Summary of Detections')
        st.plotly_chart(fig)

    st.success('Your image has been processed!')
    st.balloons()


def plot_line_between_faces(fig, ax, f1, f2, text=None, color='blue'):
    x = [f1[0] + f1[2] / 2, f2[0] + f2[2] / 2]
    y = [f1[1] + f1[3] / 2, f2[1] + f2[3] / 2]
    ax.plot(x, y, c=color)

    if text is not None:
        ax.text(np.mean(x), np.mean(y), text)


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy


def distance_to_face(face):
    focal_length = 200
    avg_face_width = 150
    avg_face_height = 0.65
    return (focal_length * avg_face_width / face[2]) / 304.8


def distance_between_faces(f1, f2):
    deltax = abs(f1[0] - f2[0])
    avg_face_width = 0.5
    avg_face_width_pixels = (f1[2] + f2[2] / 2)
    horizontal_distance = deltax * avg_face_width / avg_face_width_pixels
    vertical_distance = abs(distance_to_face(f1) - distance_to_face(f2))
    return (horizontal_distance ** 2 + vertical_distance ** 2) ** (0.5)


def overlap(b1, b2):
    from collections import namedtuple
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    ra = Rectangle(b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3])
    rb = Rectangle(b2[0], b2[1], b2[0] + b2[2], b2[1] + b2[3])
    ret = area(ra, rb)
    return ret if ret is not None else 0


def matching(masks, faces, threshold=0.5):
    faces = copy.deepcopy(faces)
    matches = []
    for mask in copy.deepcopy(masks):
        intersection = [overlap(mask, face) for face in faces]
        if len(intersection) > 0:
            best_match = np.argsort(intersection)[-1]
            if intersection[best_match] > threshold * mask[2] * mask[3]:
                matches.append(faces.pop(best_match))
    return faces, matches


def plot_faces_annotated(fig, ax, labels, color='b'):
    for box in labels:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)


def plot_masks_annotated(fig, ax, annotation, color='r'):
    for label, box in zip(annotation["labels"], annotation["boxes"]):
        if label != 1:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)
