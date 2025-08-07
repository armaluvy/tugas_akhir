import pandas as pd
import ast

# Load dataset
df = pd.read_csv("dataset_3z_labeled.csv")

# Konversi string ke list Python
df["input_ids"] = df["input_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df["attention_mask"] = df["attention_mask"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Cek apakah ada yang gagal
print("Jumlah input_ids null:", df["input_ids"].isnull().sum())
print("Jumlah attention_mask null:", df["attention_mask"].isnull().sum())

# Cek panjang sequence
print("Panjang unik input_ids:", df["input_ids"].dropna().apply(len).unique())
print("Panjang unik attention_mask:", df["attention_mask"].dropna().apply(len).unique())

# Cek distribusi label
df["labels"] = df["labels"].astype(int)
print("Distribusi label:\n", df["labels"].value_counts())

df.to_pickle("dataset_3z_ready.pkl") 