#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <climits>

// calc total cost of processing tokens w/ fixed size batching
long long fixedSizeBatching(const std::vector<int>& tokens, int batchSize, int overhead) {
    long long totCost = 0;

    // chunks of data
    for (int i = 0; i < tokens.size(); i += batchSize) {
        int maxTokens = 0;
        // handle out of bounds
        int end = std::min(i + batchSize, (int)tokens.size());
        // figure out padding needed
        for (int j = i; j < end; j++) {
            maxTokens = std::max(maxTokens, tokens[j]);
        }
        int size = end - i;
        totCost += overhead + 1LL * size * maxTokens;
    }
    return totCost;
}

long long greedyBatching(const std::vector<int>& tokens, int maxBatchSize, int overhead) {
    if (tokens.empty()) {
        return 0;
    }

    long long totCost = 0;
    int size = 1;
    int maxTokens = tokens[0];

    for (int i = 1; i < (int)tokens.size(); i++) {
        // update max token size
        int newMax = std::max(tokens[i], maxTokens);
        // costs of adding to cur batch vs starting new one
        long long addCost = (size + 1) * newMax - size*maxTokens;
        long long newBatchCost = overhead + tokens[i];

        // greedy choice of keeping cur batch if cheaper
        if (size < maxBatchSize && addCost <= newBatchCost) {
            size++;
            maxTokens = newMax;
        }
        else {
            // otherwise finish cur batch and start new one
            totCost += overhead + 1LL * size * maxTokens;
            size = 1;
            maxTokens = tokens[i];
        }
    }

    // add in final batch
    totCost += overhead + 1LL * size * maxTokens;
    return totCost;
}

long long optimalBatching(const std::vector<int>& tokens, int maxBatchSize, int overhead) {
    if (tokens.empty()) {
        return 0;
    }
    int n = (int)tokens.size();

    // dp[i] = min cost to do first i
    std::vector<long long> dp(n + 1, LLONG_MAX);
    dp[0] = 0;

    // calc optimal cost for each prefix
    for (int i = 0; i <= n; i++) {
        int maxTokens = 0;
        // iterate till batch full or out of bounds
        for (int j = i - 1; j >= std::max(0, i - maxBatchSize); j--) {
            // update longest token size
            maxTokens = std::max(maxTokens, tokens[j]);
            // compare going this far back to cur best
            int size = i - j;
            long long batchCost = overhead + 1LL * size * maxTokens;
            long long totCost = batchCost + dp[j];
            dp[i] = std::min(dp[i], totCost);
        }
    }
    return dp[n];
}

int main() {
    std::ifstream file("data/request_tokens_1000.txt");

    if (!file.is_open()) {
        std::cerr << "Error opening file\n";
        return 1;
    }

    std::vector<int> tokens;
    int count;

    while (file >> count) {
        tokens.push_back(count);
    }
    std::cout << "Loaded " << tokens.size() << " requests\n";

    int batchSize = 8;
    int overhead = 50;

    std::cout << "Fixed-size batching cost: "
            << fixedSizeBatching(tokens, batchSize, overhead)
            << "\n";

    std::cout << "Greedy batching cost: "
            << greedyBatching(tokens, batchSize, overhead)
            << "\n";

    // compare DP against existing algos
    std::cout << "Optimal DP batching cost: "
            << optimalBatching(tokens, batchSize, overhead)
            << "\n";

    // verify on small ex
    std::vector<int> testTokens = {10, 12, 100, 105, 11, 13};

    std::cout << "Small DP test: "
            << optimalBatching(testTokens, 3, 50)
            << "\n"; // expected to be 410
    
    return 0;
}
