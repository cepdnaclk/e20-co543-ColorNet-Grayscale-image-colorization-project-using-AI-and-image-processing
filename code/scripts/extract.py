import cv2
import os
import numpy as np

def resize_with_padding(image, target_size=(144, 256)):
    h, w = image.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)  
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left
    
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image

def extract_frames(videos, output_dir, frame_interval=800):
    """
    Extract frames from multiple videos at a fixed interval, resize them, and save.

    :param videos: List of video file paths.
    :param output_dir: Directory to save the extracted frames.
    :param frame_interval: Interval between frames to extract.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_saved_count = 0  # Keeps unique numbering across videos

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Couldn't open video file {video_path}")
            continue

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # No more frames left in the video, move to the next one

            if frame_count % frame_interval == 0:
                print(f"Extracting frame {frame_count} from {video_path}")
                resized_frame = resize_with_padding(frame, (144, 256))
                frame_filename = os.path.join(output_dir, f"frame_{total_saved_count:05d}.jpg")
                cv2.imwrite(frame_filename, resized_frame)
                total_saved_count += 1

            frame_count += 1
        
        cap.release()
        print(f"Processed {video_path}, extracted {total_saved_count} frames so far.")

def convert_to_lab_and_save(input_dir, output_l_dir, output_ab_dir):
    os.makedirs(output_l_dir, exist_ok=True)
    os.makedirs(output_ab_dir, exist_ok=True)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)

        if img_file.endswith('.jpg'):
            img = cv2.imread(img_path)
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab_image)

            L_normalized = L / 255.0
            a_normalized = a / 128.0 - 1
            b_normalized = b / 128.0 - 1

            frame_name = img_file.replace('.jpg', '')
            L_filename = os.path.join(output_l_dir, f"{frame_name}_L.jpg")
            cv2.imwrite(L_filename, (L_normalized * 255).astype(np.uint8))

            a_filename = os.path.join(output_ab_dir, f"{frame_name}_a.jpg")
            b_filename = os.path.join(output_ab_dir, f"{frame_name}_b.jpg")
            cv2.imwrite(a_filename, ((a_normalized + 1) * 128).astype(np.uint8))
            cv2.imwrite(b_filename, ((b_normalized + 1) * 128).astype(np.uint8))


# Example usage:
videos = [
    "E:\\sem 5\\DIP\\mini_pro\\CO543_mini_project\\scripts\\V1.mkv"
]  # Add more video paths as needed

output_frames_dir = 'E:\\sem 5\\DIP\\mini_pro\\CO543_mini_project\\dataset\\frames'
output_l_dir = 'E:\\sem 5\\DIP\\mini_pro\\CO543_mini_project\\dataset\\processed\\L'
output_ab_dir = 'E:\\sem 5\\DIP\\mini_pro\\CO543_mini_project\\dataset\\processed\\ab'

extract_frames(videos, output_frames_dir, frame_interval=300)
convert_to_lab_and_save(output_frames_dir, output_l_dir, output_ab_dir)
