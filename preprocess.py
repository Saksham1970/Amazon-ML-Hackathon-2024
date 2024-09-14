import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp

class ImageProcessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) / 9

    def enhance_image(self, image):
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        enhanced = self.clahe.apply(gray)
        
        # Quick sharpening
        sharpened = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
        
        return sharpened

    def resize_image(self, image, max_size=600): 
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

def process_image(args):
    input_path, output_path = args
    processor = ImageProcessor()
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Could not load image {input_path}. Skipping.")
        return input_path

    resized = processor.resize_image(image)
    enhanced = processor.enhance_image(resized)
    cv2.imwrite(output_path, enhanced)
    return input_path

def main():
    input_folder = "./dataset/train"
    output_folder = "./processed/train"
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    args_list = [(os.path.join(input_folder, f), 
                  os.path.join(output_folder, f)) 
                 for f in image_files]
    
   

    num_processes = mp.cpu_count()
    
    with mp.Pool(num_processes) as pool:
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for _ in pool.imap_unordered(process_image, args_list, chunksize=20):
                pbar.update()

if __name__ == "__main__":
    main()