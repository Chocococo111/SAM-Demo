import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import cv2

# install the packages first 
# pip install torch torchvision
# pip install segment-anything
# pip install opencv-python matplotlib

    # download SAM model using wget or from the official website
    # !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

def load_sam_model():
    # Default model is 'vit_h' can use 'vit_b' for a smaller, faster model

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    return sam

"""
Set up the predictor with an image
"""
def setup_predictor(sam_model, image):

    predictor = SamPredictor(sam_model)
    predictor.set_image(image)
    return predictor

"""
Generate mask from a point prompt
"""
def segment_from_point(predictor, point_coords, point_labels):
    # masks from points
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # return mask with highest score
    mask_idx = np.argmax(scores)
    return masks[mask_idx]

"""
Visualize the original image and the segmentation mask
"""
def visualize_results(image, mask):
  
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmentation mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.7, cmap='jet')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.show()

def main():
    # sample image
    image_path = "sample_medical_image.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # SAM model
    print("Loading SAM model...")
    sam_model = load_sam_model()
    
    # Setup predictor
    print("Setting up predictor...")
    predictor = setup_predictor(sam_model, image)
    
    # click in the center of the image
    input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
    input_label = np.array([1])  # 1 indicates a foreground point
    
    # Generate mask
    print("Generating segmentation...")
    mask = segment_from_point(predictor, input_point, input_label)
    
    print("Displaying results...")
    visualize_results(image, mask)

if __name__ == "__main__":
    main()