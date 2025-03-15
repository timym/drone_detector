import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import argparse

# Custom dataset class for loading drone images and annotations
class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, root, images_folder="images", annotations_folder="annotations", transforms=None):
        self.root = root
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.transforms = transforms
        # List all image files in the specified images subfolder
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.images_folder))))

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, self.images_folder, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Load corresponding annotation file (same basename with .txt extension)
        annot_path = os.path.join(self.root, self.annotations_folder, os.path.splitext(self.imgs[idx])[0] + ".txt")
        boxes = []
        with open(annot_path, "r") as f:
            for line in f:
                # Each line should contain: xmin ymin xmax ymax
                coords = list(map(float, line.strip().split()))
                boxes.append(coords)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # All objects are assumed to be "drone" (class label 1); background is implicitly 0
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# Function to load a pre-trained Faster R-CNN model and adapt it for drone detection
def get_model(num_classes):
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (2 classes: background and drone)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(dataset_path, images_folder="images", annotations_folder="annotations"):
    # Create the dataset and data loader using the provided paths
    dataset = DroneDataset(dataset_path, images_folder=images_folder,
                           annotations_folder=annotations_folder,
                           transforms=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(num_classes=2)  # 2 classes: background and drone
    model.to(device)

    # Define an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: compute losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {losses.item()}")

    # Save the trained model weights
    torch.save(model.state_dict(), "drone_detector.pth")
    print("Training complete. Model saved as drone_detector.pth.")

def main():
    parser = argparse.ArgumentParser(description="Train a Drone Detector model using a custom dataset.")
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset root folder containing images and annotations subfolders.")
    parser.add_argument("--images", default="images",
                        help="Name of the images subfolder inside the dataset folder (default: images).")
    parser.add_argument("--annotations", default="annotations",
                        help="Name of the annotations subfolder inside the dataset folder (default: annotations).")
    args = parser.parse_args()

    # Call the training function with the provided dataset and folder names
    train_model(dataset_path=args.dataset, images_folder=args.images, annotations_folder=args.annotations)

if __name__ == "__main__":
    main()
