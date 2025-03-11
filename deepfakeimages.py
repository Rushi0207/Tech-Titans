import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import warnings
warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def predict(input_image: Image.Image):
    """Predict the label of the input_image"""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)  # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    # Convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    # Select target layer for Grad-CAM (use an appropriate layer for your model)
    target_layer = model.block8.branch1[-1]

    grad_cam = LayerGradCam(model, target_layer)
    
    # Get the attribution for the input image (Grad-CAM output)
    attr = grad_cam.attribute(face, target=0)
    
    # Visualize the attribution and apply it on the image
    visualization = viz.visualize_image_attr(attr[0].cpu().detach().numpy(), img=prev_face, method="heat_map", sign="all")

    # Combine the heatmap with the original image
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    
    return confidences, face_with_mask

interface = gr.Interface(
    fn=predict,
    inputs=[gr.inputs.Image(label="Input Image", type="pil")],
    outputs=[gr.outputs.Label(label="Class"),
             gr.outputs.Image(label="Face with Explainability", type="pil")],
).launch()
