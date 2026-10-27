#define BATCHING_TEST
#include "../src/main.cpp"

#include <cassert>
#include <climits>
#include <iostream>

// brute force to independently verify optimal DP cost
long long bruteForce(const std::vector<int>& tokens, int maxBatchSize, int overhead, int start = 0) {
    if (start == (int)tokens.size()) {
        return 0;
    }

    long long minCost = LLONG_MAX;
    int maxTokens = 0;

    for (int i = start; i < std::min(start + maxBatchSize, (int)tokens.size()); i++) {
        maxTokens = std::max(maxTokens, tokens[i]);
        int size = i - start + 1;
        long long batchCost = overhead + 1LL * size * maxTokens;
        long long totCost = batchCost + bruteForce(tokens, maxBatchSize, overhead, i + 1);
        minCost = std::min(minCost, totCost);
    }
    return minCost;
}

void testEmpty() {
    std::vector<int> tokens;
    assert(fixedSizeBatching(tokens, 3, 50) == 0);
    assert(greedyBatching(tokens, 3, 50) == 0);
    assert(optimalBatching(tokens, 3, 50) == 0);
}

void testSingleRequest() {
    std::vector<int> tokens = {20};
    assert(fixedSizeBatching(tokens, 3, 50) == 70);
    assert(greedyBatching(tokens, 3, 50) == 70);
    assert(optimalBatching(tokens, 3, 50) == 70);
}

void testKnownExample() {
    std::vector<int> tokens = {10, 12, 100, 105, 11, 13};
    assert(fixedSizeBatching(tokens, 3, 50) == 715);
    assert(greedyBatching(tokens, 3, 50) == 410);
    assert(optimalBatching(tokens, 3, 50) == 410);
}

void testBatchSizeOne() {
    std::vector<int> tokens = {10, 12, 100};
    assert(fixedSizeBatching(tokens, 1, 50) == 272);
    assert(greedyBatching(tokens, 1, 50) == 272);
    assert(optimalBatching(tokens, 1, 50) == 272);
}

void testAgainstBruteForce() {
    std::vector<std::vector<int>> samples = {
        {10, 12, 100, 105, 11, 13},
        {5, 100, 10, 80, 20},
        {50, 50, 50, 50},
        {1, 2, 3, 4, 5}
    };

    for (const auto& tokens : samples) {
        for (int batchSize = 1; batchSize <= 3; batchSize++) {
            for (int overhead : {0, 50, 200}) {
                long long dpCost = optimalBatching(tokens, batchSize, overhead);
                long long bruteCost = bruteForce(tokens, batchSize, overhead);
                assert(dpCost == bruteCost);
                assert(dpCost <= greedyBatching(tokens, batchSize, overhead));
                assert(dpCost <= fixedSizeBatching(tokens, batchSize, overhead));
            }
        }
    }
}

int main() {
    testEmpty();
    testSingleRequest();
    testKnownExample();
    testBatchSizeOne();
    testAgainstBruteForce();
    std::cout << "All batching tests passed!\n";
    return 0;
}
