# Amazon ML Challenge 2024: Entity Value Extraction from Images

## Overview

This repository contains our solution for the **Amazon ML Challenge 2024**, where we ranked **34th** with an **F1 score of 0.642**. The challenge focused on extracting entity values (e.g., weight, volume, dimensions) directly from product images, a critical task for e-commerce platforms to ensure comprehensive product information.

---

## Problem Statement

The goal was to develop a machine learning model capable of extracting entity values directly from product images. This capability addresses gaps where textual product descriptions are missing or incomplete, enabling extraction of vital information such as weight, volume, voltage, wattage, and dimensions.

---

## Dataset and Output Format

### Input:
- **Training Data**: Images with labeled entities (e.g., weight, height).
- **Test Data**: Images without labeled entities; the model must predict entity values.

### Output:
A CSV file with the following format:
- **index**: Unique identifier for each data sample.
- **prediction**: Predicted entity value in the format `"x unit"`, where `x` is a float, and `unit` is one of the allowed units.

---

## Approach

### Image Preprocessing:
1. **Resizing**: 
   - Images resized to a maximum of 600 pixels to maintain aspect ratio while reducing computational overhead.
2. **Contrast Enhancement (CLAHE)**:
   - Improved visibility of darker regions for better text readability.
3. **Sharpening (Unsharp Masking)**:
   - Enhanced fine details such as edges and labels for precise entity extraction.

### Model Architecture:
- **Model Used**: **IDEFICS2** (Vision-Language Model)
  - Pretrained for tasks like Optical Character Recognition (OCR) and Visual Question Answering (VQA).
  - Extracted textual details directly from images.
  - Ideal for handling scenarios with limited or unclear textual descriptions in images.

### Fine-tuning:
- **Technique**: QLoRA (Quantized Low-Rank Adapter)
  - Efficient fine-tuning by updating specific model layers with limited computational resources.
- **Dataset**: 1.5 lakh images from the training set.
- **Epochs**: Single epoch with quantized (16-bit floating point) weights.

---

## Pipeline

### 1. **Preprocessing**:
- Script: `preprocess.py`
- Key Features:
  - Batch processing of images using multiprocessing.
  - Image enhancement with CLAHE and sharpening.
  - Resized images stored for model training.

### 2. **Model Training**:
- Notebook: `training.ipynb`
- Key Features:
  - Data formatted with a custom data collator.
  - QLoRA fine-tuning of IDEFICS2 using the Hugging Face library.
  - Training configuration optimized for entity extraction.

### 3. **Inference**:
- Notebook: `inference.ipynb`
- Key Features:
  - Trained model applied to test images.
  - Predictions saved as a CSV file in the required format.

---

## Results

- **F1 Score**: **0.642**
- **Rank**: **34th**

The solution successfully extracted entity values with competitive accuracy, leveraging efficient preprocessing, a fine-tuned Vision-Language Model, and optimized inference.

---

## Instructions to Run

### 1. Preprocessing:
```bash
python preprocess.py
```

### 2. Model Training:
Open and execute `training.ipynb` to fine-tune the IDEFICS2 model.

### 3. Inference:
Open and execute `inference.ipynb` to generate predictions on the test data.

### 4. Output Validation:
Use `src/sanity.py` to ensure the output file format matches the requirements:
```bash
python src/sanity.py --file path_to_test_out.csv
```

---

## Technologies Used

- **Frameworks**: Hugging Face Transformers, PyTorch
- **Libraries**: OpenCV, Pandas, tqdm
- **Language**: Python

---

## Lessons Learned

This challenge highlighted the importance of:
- Effective image preprocessing for OCR tasks.
- Efficient model fine-tuning techniques like QLoRA for resource optimization.
- Robust pipeline development for end-to-end ML workflows.

---

Feel free to reach out for more details about the implementation or potential improvements!
