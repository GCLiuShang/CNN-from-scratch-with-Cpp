#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
// Softmax + 交叉熵损失：
// - 作用：把模型输出的未归一化分数 logits 转为概率，再用交叉熵衡量与真实标签分布的差异。
// - 数值稳定：计算 softmax 时先减去每个样本的最大值，避免 exp 溢出。
// - 前向：loss = 平均(-log p_true)；其中 p_true 为真实类别对应的 softmax 概率。
// - 反向：对每个样本的 logits 梯度为 (p - one_hot(label)) / batch。
struct SoftmaxCE {
    int classes, batch_cache;           // 类别数，缓存批大小
    std::vector<int> y_true;            // 标签缓存
    std::vector<float> probs;           // softmax 概率缓存
    SoftmaxCE(int classes_) : classes(classes_) {}
    float forward(const std::vector<float>& logits, const std::vector<int>& y, int batch) {
        batch_cache = batch; y_true = y; probs.assign((size_t)batch * classes, 0.0f); float loss = 0.0f;
        for (int n = 0; n < batch; ++n) {
            float mx = -1e9f; for (int c = 0; c < classes; ++c) mx = std::max(mx, logits[(size_t)n*classes + c]);
            float sumexp = 0.0f; for (int c = 0; c < classes; ++c) sumexp += expf(logits[(size_t)n*classes + c] - mx);
            for (int c = 0; c < classes; ++c) probs[(size_t)n*classes + c] = expf(logits[(size_t)n*classes + c] - mx) / sumexp;
            loss += -logf(std::max(probs[(size_t)n*classes + y[n]], 1e-12f));
        }
        return loss / batch;
    }
    std::vector<float> backward() {
        std::vector<float> dlogits = probs;
        for (int n = 0; n < batch_cache; ++n) dlogits[(size_t)n*classes + y_true[n]] -= 1.0f;
        for (float &v : dlogits) v /= batch_cache;
        return dlogits;
    }
};
