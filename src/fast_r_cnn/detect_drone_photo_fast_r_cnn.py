import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse

def get_model(num_classes):
    """
    Load a pre-trained Faster R-CNN model and replace its head to predict `num_classes` (background + drone).
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (num_classes includes background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes=2):
    """
    Load the trained model weights from disk.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def detect_drones(image_path, model, threshold=0.5):
    """
    Detect drones in an image, drawing bounding boxes around detections with a score above the threshold.
    """
    device = next(model.parameters()).device
    transform = transforms.ToTensor()
    
    # Load image and convert to tensor
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Convert PIL image to OpenCV format for drawing (RGB to BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process detections (assuming label 1 is "drone")
    output = outputs[0]
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        if score > threshold and label == 1:
            # Convert box coordinates to integers
            x_min, y_min, x_max, y_max = box.int().tolist()
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_cv, f"Drone: {score:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_cv

def main():
    parser = argparse.ArgumentParser(description="Drone Detection from an Image using a trained model.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model weights (e.g., drone_detector.pth)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file for drone detection")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (default: 0.5)")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)
    
    # Detect drones in the image
    result_image = detect_drones(args.image, model, threshold=args.threshold)
    
    # Display the result
    cv2.imshow("Drone Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
