import pandas as pd
import re

# ---- Automatically find the first CSV file in your folder ----
import glob
files = glob.glob("*.csv")
if not files:
    raise FileNotFoundError("❌ No CSV files found in this folder. Please place your dataset here.")
filename = files[0]
print(f"📂 Found file: {filename}")

# ---- Load dataset ----
df = pd.read_csv(filename)
print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

# ---- Try to detect resume text columns ----
resume_cols = [c for c in df.columns if 'resume' in c.lower()]
if not resume_cols:
    raise KeyError("❌ No columns containing 'resume' found in this CSV.")

# ---- Combine all resume text columns into one ----
df['Resume'] = df[resume_cols].fillna('').astype(str).agg(' '.join, axis=1)

# ---- Clean HTML tags or special characters ----
df['Resume'] = df['Resume'].apply(lambda x: re.sub(r'<[^>]+>', ' ', str(x)))
df['Resume'] = df['Resume'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# ---- Detect or create Category column ----
if 'Category' not in df.columns:
    print("⚠️  No 'Category' column found — assigning 'Unknown' for all rows.")
    df['Category'] = 'Unknown'

# ---- Keep only relevant columns ----
df = df[['Resume', 'Category']]

# ---- Remove blank resumes ----
df = df[df['Resume'].str.strip() != '']
df.to_csv("resume_clean.csv", index=False)

print(f"✅ Cleaning complete! Saved as resume_clean.csv ({len(df)} rows).")
