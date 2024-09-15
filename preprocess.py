import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp

class ImageProcessor:
    def __init__(self):
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) / 9

    def enhance_image(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE only to the L-channel with reduced clip limit
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l)
        
        # Merge the channels back
        lab_enhanced = cv2.merge((enhanced_l, a, b))
        
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply reduced sharpening
        sharpened = self.unsharp_mask(enhanced)
        
        return sharpened
    
    def unsharp_mask(self, image, sigma=1.0, strength=1.5):
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
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
    os.remove(input_path)
    return input_path

def main():
    input_folder = "./images/train/train"
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
