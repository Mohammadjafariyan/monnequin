import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

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
    person_mask = output_predictions == 15  # Assuming class 15 is the person
    hair_mask = output_predictions == 17  # Assuming class 17 is the hair
    return person_mask, hair_mask

def detect_and_mask_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_mask = np.zeros_like(gray_image, dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x+w, y+h), 255, -1)  # White rectangle mask on the face area
    return face_mask

def inpaint_image(image, mask):
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_image

def set_mannequin_color(image, person_mask, color=(255, 255, 255)):
    mannequin_image = image.copy()
    mannequin_image[person_mask] = color  # Set the color of the person region
    return mannequin_image

# Load image
image_path = 'D:/Temp/clothes/wedding.jpg'
image = load_image(image_path)

# Segment person and hair
person_mask, hair_mask = segment_person(image)

# Detect and mask the face
face_mask = detect_and_mask_face(image)

# Combine face and hair masks
combined_mask = np.maximum(face_mask, hair_mask.astype(np.uint8) * 255)

# Inpaint the face and hair areas
image_without_face_and_hair = inpaint_image(image, combined_mask)

# Set the mannequin color
mannequin_color = (255, 255, 255)  # White color for the mannequin
final_image = set_mannequin_color(image_without_face_and_hair, person_mask, mannequin_color)

# Save or display the result
final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('D:/Temp/clothes/final_image.jpg', final_image_bgr)
