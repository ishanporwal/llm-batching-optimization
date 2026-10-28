import csv
from pathlib import Path

data_dir = Path("data").resolve()
input_file = data_dir / "BurstGPT_1.csv"
output_file = data_dir / "request_tokens_10000.txt"

tokens = []
model = "ChatGPT"
numRequests = 10000

# read csv and extract first 10k valid requests for one model
with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # only use requests from selected model
        if row["Model"].strip() != model:
            continue
        try:
            count = int(float(row["Request tokens"]))
        except (ValueError, TypeError):
            continue
        if count > 0:
            tokens.append(count)
        if len(tokens) == numRequests:
            break

# save request lengths in original order
with open(output_file, "w") as f:
    for count in tokens:
        f.write(f"{count}\n")

print(f"Model: {model}")
print(f"Extracted {len(tokens)} requests")
print(f"First 10 token lengths: {tokens[:10]}")
print(f"Saved to {output_file}")
