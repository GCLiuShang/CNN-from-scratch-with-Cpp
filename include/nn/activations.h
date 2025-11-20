#pragma once
#include <vector>
// 激活函数 ReLU：
// - 定义：y = max(0, x)，将负值截断为 0，保留正值；简洁高效，缓解梯度消失。
// - 形状：这里的实现以一维向量处理（配合示例中把 Tensor4D 展开后传入）。
// - 反向：若前向中 x>0，则梯度原样传递；否则梯度为 0。为此在前向阶段用 mask 记录 x>0 的位置。
struct ReLU {
    std::vector<float> mask; // 前向中记录每个元素是否激活
    std::vector<float> forward(const std::vector<float>& x) {
        mask.resize(x.size());
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); ++i) { y[i] = x[i] > 0 ? x[i] : 0.0f; mask[i] = x[i] > 0 ? 1.0f : 0.0f; }
        return y;
    }
    std::vector<float> backward(const std::vector<float>& dy) {
        std::vector<float> dx(dy.size());
        for (size_t i = 0; i < dy.size(); ++i) dx[i] = dy[i] * mask[i];
        return dx;
    }
};
