import kagglehub
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil

# progress bar for pandas operations
tqdm.pandas()

path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")
print(f"Dataset downloaded to: {path}")

# Debug
# if os.path.isdir(path):
#     print("Contents of download directory:")
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             print(f"- {os.path.join(root, file)}")

df = pd.read_csv(f"{path}/Amazon-Products.csv")

# print column names to see what we're working with
print("\ncolumns:")
print(df.columns.tolist())

# clean and preprocess the data
df = df.dropna(subset=['name', 'main_category']) # can't use these rows

# clean ratings
df['ratings'] = df['ratings'].replace(['No ratings', 'Get'], np.nan)
df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')

# clean 'no_of_ratings'
df['no_of_ratings'] = df['no_of_ratings'].str.replace(' ratings', '').str.replace(',', '')
df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')

# clean price columns
df['discount_price'] = df['discount_price'].str.replace('₹', '').str.replace(',', '')
df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '')
df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

# filter for products with some ratings, rating score above 3.5, and at least 10 ratings
filtered_df = df[(df['ratings'].notna()) & (df['ratings'] >= 3.5) & (df['no_of_ratings'] >= 10)].copy()

print("\nAfter filtering:", filtered_df.shape)

# sample 300 products, weighted by number of ratings
print("Sampling products...")
sampled_df = filtered_df.sample(
    n=300,
    weights='no_of_ratings',
    random_state=42
)

# reset index
sampled_df = sampled_df.reset_index(drop=True)

print("\nFinal dataset shape:", sampled_df.shape)

# print category distribution
print("\nMain category distribution:")
print(sampled_df['main_category'].value_counts())

# save processed dataset
output_file = 'processed_products.csv'
sampled_df.to_csv(output_file, index=False)
print(f"\nSaved processed dataset to '{output_file}'")
