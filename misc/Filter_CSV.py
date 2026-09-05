import pandas as pd

# Read Excel file, skipping the first 2 informational rows
df = pd.read_excel("PAdata.xlsx", skiprows=2)

# Keep only Traffic Citation rows
citations = df[
    df["Offense"].astype(str).str.strip() == "Maximum Speed Limits - Freeways"
]

# Save to CSV
citations.to_csv("PAcitations.csv", index=False)

print(f"Total rows: {len(df)}")
print(f"Traffic citations saved: {len(citations)}")