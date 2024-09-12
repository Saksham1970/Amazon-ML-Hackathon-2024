from src.utils import download_images
import pandas as pd

for dataset in ["train", "test"]:
    df = pd.read_csv(f"./dataset/{dataset}.csv")
    download_images(df["image_link"], f"./dataset/{dataset}/", True)
