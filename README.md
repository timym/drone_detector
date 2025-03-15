# drone_detector
code to detect drones in videos or photos

# train fast_r_cnn
python3 src/fast_r_cnn/train_drone_detection_fast_r_cnn.py --dataset /model --images /dataset/drone_photos --annotations /dataset/drone_annotations

# detect photo
python3 detect_drone_photo.py --model /model/drone_detector.pth --image <path_to_image.jpg>

# detect video
python3 detect_drone_video.py --model /model/drone_detector.pth --video <path_to_video.mp4> --wait 0.1
