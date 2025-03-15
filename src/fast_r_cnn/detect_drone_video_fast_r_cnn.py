import argparse
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time  # for timing inference

def get_model(num_classes):
    """
    Load a pre-trained Faster R-CNN model and replace its head to predict `num_classes`
    (background + drone).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
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
    model.eval()  # Set to evaluation mode
    return model

def detect_drones_in_frame(frame, model, threshold=0.5):
    """
    Process a single video frame to detect drones. Draws bounding boxes for detections
    with a score above the threshold.
    """
    device = next(model.parameters()).device
    # Convert the frame from BGR (OpenCV format) to RGB and then to a PIL image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Convert the PIL image to tensor
    transform = transforms.ToTensor()
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
    inference_time = time.time() - start_time
    # Optional: Print the inference time per frame
    print(f"Inference time: {inference_time:.2f} seconds")
    
    output = outputs[0]
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        if score > threshold and label == 1:  # label==1 for "drone"
            x_min, y_min, x_max, y_max = box.int().tolist()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Drone: {score:.2f}", (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Detect drones in video using a trained model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model weights (e.g., drone_detector.pth)")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video file, or use 0 for webcam")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (default: 0.5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output video (optional)")
    parser.add_argument("--wait", type=int, default=1,
                        help="Delay between frames in ms (default: 1). Increase if display issues occur.")
    args = parser.parse_args()

    # Load the trained model
    model = load_model(args.model, num_classes=2)

    # Open the video capture stream (file or webcam)
    video_source = int(args.video) if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video stream or file.")
        return

    # Set up video writer if output path is provided
    writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("Press 'q' to quit the video display.")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or error reading frame.")
            break

        frame_count += 1
        print(f"Processing frame: {frame_count}")
        # Process the frame to detect drones
        frame = detect_drones_in_frame(frame, model, threshold=args.threshold)

        # Display the resulting frame
        cv2.imshow("Drone Detection Video", frame)

        # Write the frame to the output video file if writer is set
        if writer is not None:
            writer.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(args.wait) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
