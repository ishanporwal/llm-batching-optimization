
#define BATCHING_TEST
#include "main.cpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

// measure avg execution time over repeated runs in microseconds
double measureTime(const std::vector<int>& tokens, int batchSize, int overhead,
                   long long (*batchingFunc)(const std::vector<int>&, int, int)) {
    int repetitions = 1000;
    volatile long long result = 0;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; i++) {
        result = batchingFunc(tokens, batchSize, overhead);
    }
    auto end = std::chrono::steady_clock::now();

    // convert tot elapsed time to avg microseconds per run
    double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
    return elapsed / repetitions;
}

int main() {
    // load 10000 ChatGPT request lengths
    std::ifstream file("data/request_tokens_10000.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open dataset\n";
        return 1;
    }

    std::vector<int> tokens;
    int count;

    while (file >> count) {
        tokens.push_back(count);
    }
    if (tokens.size() < 10000) {
        std::cerr << "Expected at least 10000 requests\n";
        return 1;
    }

    // save results separately from original benchmarks
    std::ofstream output("results/benchmark_samples.csv");
    if (!output.is_open()) {
        std::cerr << "Could not create benchmark file\n";
        return 1;
    }

    output << "sample,numRequests,batchSize,overhead,"
           << "fixedCost,greedyCost,dpCost,"
           << "fixedTimeUs,greedyTimeUs,dpTimeUs\n";

    int sampleSize = 1000;

    // test 10 consecutive samples of 1000 requests
    for (int sample = 0; sample < 10; sample++) {
        int start = sample * sampleSize;
        int end = start + sampleSize;

        std::vector<int> sampleTokens(tokens.begin() + start, tokens.begin() + end);

        // test diff max batch sizes and overheads
        for (int batchSize : {2, 4, 8, 16}) {
            for (int overhead : {50, 250, 1000}) {
                // calc modeled costs on same inputs
                long long fixedCost = fixedSizeBatching(sampleTokens, batchSize, overhead);
                long long greedyCost = greedyBatching(sampleTokens, batchSize, overhead);
                long long dpCost = optimalBatching(sampleTokens, batchSize, overhead);
                // dp should never cost more than either baseline
                if (dpCost > fixedCost || dpCost > greedyCost) {
                    std::cerr << "Invalid DP result in sample " << sample + 1 << "\n";
                    return 1;
                }
                // benchmark execution times, averaged over 100 runs
                double fixedTime = measureTime(sampleTokens, batchSize, overhead, fixedSizeBatching);
                double greedyTime = measureTime(sampleTokens, batchSize, overhead, greedyBatching);
                double dpTime = measureTime(sampleTokens, batchSize, overhead, optimalBatching);
                // save one row for each experiment
                output <<   sample + 1    << ","
                       <<   sampleSize    << ","
                       <<   batchSize     << ","
                       <<   overhead      << ","
                       <<   fixedCost     << ","
                       <<   greedyCost    << ","
                       <<   dpCost        << ","
                       <<   fixedTime     << ","
                       <<   greedyTime    << ","
                       <<   dpTime        << "\n";
            }
        }
        std::cout << "Completed sample " << sample + 1 << "/10\n";
    }
    std::cout << "Benchmark complete! Results saved to results/benchmark_samples.csv\n";
    return 0;
}
