import cv2
import os
import sys
import glob

# Global variables used for annotation on the current image
refPt = []      # To store the two points (start and end) of the current box
boxes = []      # To store all drawn boxes on the current image
cropping = False

def click_and_crop(event, x, y, flags, param):
    """
    Mouse callback function to record points and draw rectangles.
    """
    global refPt, cropping, boxes, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a new rectangle
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        # While dragging, display the rectangle dynamically
        temp_image = clone.copy()
        cv2.rectangle(temp_image, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finalize the rectangle when the left mouse button is released
        refPt.append((x, y))
        cropping = False
        boxes.append(refPt)
        cv2.rectangle(clone, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", clone)

def annotate_images_in_folder(folder_path):
    """
    Iterates over images in the given folder and allows the user to annotate each.
    The annotation for each image is saved as a text file with the same basename.
    """
    # Supported image extensions
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    if not image_files:
        print("No images found in folder:", folder_path)
        return

    image_files.sort()  # For consistency

    for image_path in image_files:
        print("\nAnnotating image:", image_path)
        # Load the image and set up the working copy
        image = cv2.imread(image_path)
        if image is None:
            print("Error loading image:", image_path)
            continue

        # Reset global variables for the new image
        global clone, boxes, refPt, cropping
        clone = image.copy()
        boxes = []
        refPt = []
        cropping = False

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", click_and_crop)

        print("Instructions:")
        print(" - Click and drag to draw a bounding box.")
        print(" - Press 'r' to reset annotations for this image.")
        print(" - Press 'n' to save annotations and move to the next image.")
        print(" - Press 'q' to quit the annotation tool.")

        # Annotation loop for the current image
        while True:
            cv2.imshow("Image", clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                # Reset current annotations
                clone = image.copy()
                boxes = []
                print("Annotations reset for this image.")

            elif key == ord("n"):
                # Save annotations for the current image and break out to process the next one
                if boxes:
                    annotation_file = os.path.splitext(image_path)[0] + ".txt"
                    with open(annotation_file, "w") as f:
                        for box in boxes:
                            # Ensure (xmin, ymin, xmax, ymax) ordering
                            x_min = min(box[0][0], box[1][0])
                            y_min = min(box[0][1], box[1][1])
                            x_max = max(box[0][0], box[1][0])
                            y_max = max(box[0][1], box[1][1])
                            f.write(f"{x_min} {y_min} {x_max} {y_max}\n")
                    print(f"Annotations saved to {annotation_file}")
                else:
                    print("No annotations were made for this image.")
                break

            elif key == ord("q"):
                # Quit the tool completely
                print("Exiting annotation tool.")
                cv2.destroyAllWindows()
                return

        # Close the window for the current image before moving on
        cv2.destroyWindow("Image")

def main():
    if len(sys.argv) < 2:
        print("Usage: python annotate_folder.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print("The provided folder path does not exist:", folder_path)
        sys.exit(1)

    annotate_images_in_folder(folder_path)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
