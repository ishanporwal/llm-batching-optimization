import csv
from pathlib import Path
import matplotlib.pyplot as plt

resultsDir = Path(__file__).resolve().parent.parent / "results"
inputFile = resultsDir / "benchmark_samples.csv"

# load benchmark results
data = []
with open(inputFile, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            key: float(value) if key.endswith("TimeUs") else int(value)
            for key, value in row.items()
        })

# avg vals across samples
def average(rows, key):
    return sum(row[key] for row in rows) / len(rows)


# Graph 1 -> effect of max batch size
# keep overhead constant and average across 10 samples
batchSizes = [2, 4, 8, 16]
fixedCosts = []
greedyCosts = []
dpCosts = []
for batchSize in batchSizes:
    sample = [row for row in data if row["batchSize"] == batchSize and row["overhead"] == 250]
    fixedCosts.append(average(sample, "fixedCost"))
    greedyCosts.append(average(sample, "greedyCost"))
    dpCosts.append(average(sample, "dpCost"))

plt.figure(figsize=(8, 5))
plt.plot(batchSizes, fixedCosts, marker="o", label="Fixed-size")
plt.plot(batchSizes, greedyCosts, marker="o", label="Greedy")
plt.plot(batchSizes, dpCosts, marker="o", label="Optimal DP")
plt.xlabel("Maximum Batch Size")
plt.ylabel("Average Modeled Processing Cost")
plt.title("Batch Size vs. Processing Cost (10 Samples)")
plt.xticks(batchSizes)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(resultsDir / "batch_size_comparison.png", dpi=150)
plt.close()

# Graph 2 -> effect of per-batch overhead
# keep max batch size constant and average across 10 samples
overheads = [50, 250, 1000]
fixedCosts = []
greedyCosts = []
dpCosts = []
for overhead in overheads:
    sample = [row for row in data if row["batchSize"] == 8 and row["overhead"] == overhead]
    fixedCosts.append(average(sample, "fixedCost"))
    greedyCosts.append(average(sample, "greedyCost"))
    dpCosts.append(average(sample, "dpCost"))

plt.figure(figsize=(8, 5))
plt.plot(overheads, fixedCosts, marker="o", label="Fixed-size")
plt.plot(overheads, greedyCosts, marker="o", label="Greedy")
plt.plot(overheads, dpCosts, marker="o", label="Optimal DP")
plt.xlabel("Overhead Per Batch (Cost Units)")
plt.ylabel("Average Modeled Processing Cost")
plt.title("Batch Overhead vs. Processing Cost (10 Samples)")
plt.xticks(overheads)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(resultsDir / "overhead_comparison.png", dpi=150)
plt.close()

# Graph 3 -> DP improvement over greedy across diff samples
# keep batch size at 8 and compare diff overheads
plt.figure(figsize=(8, 5))

for overhead in overheads:
    sample = [row for row in data if row["batchSize"] == 8 and row["overhead"] == overhead]
    sample.sort(key=lambda row: row["sample"])
    sampleNums = [row["sample"] for row in sample]
    improvements = [
        100 * (row["greedyCost"] - row["dpCost"]) / row["greedyCost"]
        for row in sample
    ]
    plt.plot(sampleNums, improvements, marker="o", label=f"Overhead = {overhead}")
    print(f"Overhead {overhead}: Average DP improvement = {average([{'improvement': x} for x in improvements], 'improvement'):.2f}%")

plt.xlabel("Workload Sample")
plt.ylabel("DP Improvement Over Greedy (%)")
plt.title("DP vs. Greedy Across 10 Workload Samples")
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(resultsDir / "dp_improvement.png", dpi=150)
plt.close()

# Graph 4 -> execution time vs max batch size
# keep overhead constant and avg across 10 samples
fixedTimes = []
greedyTimes = []
dpTimes = []
for batchSize in batchSizes:
    sample = [row for row in data if row["batchSize"] == batchSize and row["overhead"] == 250]
    fixedTimes.append(average(sample, "fixedTimeUs"))
    greedyTimes.append(average(sample, "greedyTimeUs"))
    dpTimes.append(average(sample, "dpTimeUs"))

plt.figure(figsize=(8, 5))
plt.plot(batchSizes, fixedTimes, marker="o", label="Fixed-size")
plt.plot(batchSizes, greedyTimes, marker="o", label="Greedy")
plt.plot(batchSizes, dpTimes, marker="o", label="Optimal DP")
plt.xlabel("Maximum Batch Size")
plt.ylabel("Average Execution Time (microseconds)")
plt.title("Algorithm Execution Time (1000 Requests)")
plt.xticks(batchSizes)
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(resultsDir / "execution_time.png", dpi=150)
plt.close()

print("All 4 graphs saved to results/")
