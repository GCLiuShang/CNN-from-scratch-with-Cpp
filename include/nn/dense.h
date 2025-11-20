#pragma once
#include <vector>
#include <random>
#include <algorithm>
// Dense 全连接层：
// - 作用：将展平后的特征映射到输出维度（如类别数）。计算公式 y = xW + b。
// - 形状：
//   输入 x 视为 [batch, in_dim]；权重 W 视为 [in_dim, out_dim]；输出 y 视为 [batch, out_dim]。
// - 前向：逐样本、逐输出维度累加（内层循环 over in_dim）。
// - 反向：
//   dW = X^T · dY（对每个 (i,o) 累加所有样本的 x[i] * dy[o]）；
//   db = sum(dY)（对每个 o 累加所有样本的 dy[o]）；
//   dX = dY · W^T（对每个 i 累加所有 o 的 W[i,o] * dy[o]）。
struct Dense {
    int in_dim, out_dim, batch_cache;           // 输入维度、输出维度、缓存的批大小
    std::vector<float> W, b, x_cache, y_cache, dW, db; // 权重、偏置、前向缓存、梯度缓存
    Dense(int in_dim_, int out_dim_) : in_dim(in_dim_), out_dim(out_dim_) {
        W.resize((size_t)in_dim * out_dim);
        b.resize((size_t)out_dim);
        std::random_device rd; std::mt19937 gen(rd()); std::normal_distribution<float> nd(0.0f, 0.1f);
        for (auto &w : W) w = nd(gen);
        for (auto &bb : b) bb = 0.0f;
        dW.assign((size_t)in_dim * out_dim, 0.0f);
        db.assign((size_t)out_dim, 0.0f);
    }
    std::vector<float> forward(const std::vector<float>& x, int batch) {
        batch_cache = batch;                // 缓存批大小用于反向归一化
        x_cache = x;                        // 缓存输入以计算 dW
        std::vector<float> y((size_t)batch * out_dim, 0.0f);
        for (int n = 0; n < batch; ++n) {
            for (int o = 0; o < out_dim; ++o) {
                float s = b[o];
                for (int i = 0; i < in_dim; ++i) s += x[(size_t)n*in_dim + i] * W[(size_t)i*out_dim + o];
                y[(size_t)n*out_dim + o] = s;
            }
        }
        y_cache = y;                        // 缓存输出以便需要时使用
        return y;
    }
    std::vector<float> backward(const std::vector<float>& dy) {
        std::fill(dW.begin(), dW.end(), 0.0f);
        std::fill(db.begin(), db.end(), 0.0f);
        std::vector<float> dx((size_t)batch_cache * in_dim, 0.0f);
        for (int n = 0; n < batch_cache; ++n) {
            for (int o = 0; o < out_dim; ++o) {
                float g = dy[(size_t)n*out_dim + o];   // dY 的梯度
                db[o] += g;                             // 偏置梯度为 dY 的累加
                for (int i = 0; i < in_dim; ++i) {
                    dW[(size_t)i*out_dim + o] += x_cache[(size_t)n*in_dim + i] * g; // dW = X^T · dY
                    dx[(size_t)n*in_dim + i] += W[(size_t)i*out_dim + o] * g;       // dX = dY · W^T
                }
            }
        }
        return dx;
    }
    void update(float lr) {                 // SGD 权重更新
        for (int i = 0; i < in_dim; ++i) for (int o = 0; o < out_dim; ++o) W[(size_t)i*out_dim + o] -= lr * dW[(size_t)i*out_dim + o];
        for (int o = 0; o < out_dim; ++o) b[o] -= lr * db[o];
    }
};
