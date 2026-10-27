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

    // parent[i] = start idx of last batch for first i requests
    std::vector<int> parent(n + 1, -1);

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
            // if partition = lower cost, save
            if (totCost < dp[i]) {
                dp[i] = totCost;
                parent[i] = j;
            }
        }
    }

    // reverse through to gather batches
    std::vector<std::pair<int,int>> batches;
    for (int i = n; i > 0; i = parent[i]) {
        batches.push_back({parent[i], i});
    }
    std::reverse(batches.begin(), batches.end());

    #ifndef BATCHING_TEST
    std::cout <<"DP formed " << batches.size() << " batches\n";

    // print first 5 batches
    int show = std::min(5, (int)batches.size());
    for (int i = 0; i < show; i++) {
        std::cout <<"Batch " << i + 1 << ": ";
        for (int j = batches[i].first; j < batches[i].second; j++) {
            std::cout << tokens[j] << " ";
        }
        std::cout << "\n";
    }
    #endif
    return dp[n];
}

#ifndef BATCHING_TEST
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

    // compare dp against existing algos
    long long dpCost = optimalBatching(tokens, batchSize, overhead);
    std::cout << "Optimal DP batching cost: " << dpCost << "\n";

    // verify dp on small ex
    std::vector<int> testTokens = {10, 12, 100, 105, 11, 13};
    long long testCost = optimalBatching(testTokens, 3, 50);
    std::cout << "Small DP test: " << testCost << "\n"; // 410 expected
    
    return 0;
}
#endif
