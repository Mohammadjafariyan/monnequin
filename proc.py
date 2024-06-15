import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import random
import string


def load_image(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def segment_person(image):
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = output_predictions == 15  # Assuming class 15 is the person
    return mask

def hide_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    mask = np.zeros_like(image, dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)  # White rectangle mask on the face area

    hidden_face_image = np.where(mask == 255, np.zeros_like(image), image)  # Replace face region with black
    return hidden_face_image

def replace_with_mannequin(image, mask, mannequin_parts_path):
    mannequin_parts = cv2.imread(mannequin_parts_path)
    mannequin_image = np.zeros_like(image)
    mannequin_image[mask] = mannequin_parts[mask]
    final_image = np.where(mask[..., None], mannequin_image, image)
    return final_image


val = input("Enter file name: ") 

# Load image
image_path = 'D:/Temp/clothes/' + val
image = load_image(image_path)

# Segment person
mask = segment_person(image)

# Hide the face
image_with_hidden_face = hide_face(image)

# Replace body parts with mannequin parts
mannequin_parts_path = 'D:/Temp/clothes/'+val
final_image = replace_with_mannequin(image_with_hidden_face, mask, mannequin_parts_path)

# Save or display the result
final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
char_set = string.ascii_uppercase + string.digits
name= ''.join(random.sample(char_set*6, 6))

cv2.imwrite('path_to_save_'+name+'.jpg', final_image)
