import csv
from pathlib import Path

data_dir = Path("data").resolve()
input_file = data_dir / "BurstGPT_1.csv"
output_file = data_dir / "request_tokens_1000.txt"

tokens = []

# read csv and extract first 1000 valid request token lengths
with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            count = int(float(row["Request tokens"]))
        except (ValueError, TypeError):
            continue
        if count > 0:
            tokens.append(count)
        if len(tokens) == 1000:
            break

with open(output_file, "w") as f:
    for count in tokens:
        f.write(f"{count}\n")

print(f"Extracted {len(tokens)} requests")
print(f"First 10 token lengths: {tokens[:10]}")
print(f"Saved to {output_file}")
