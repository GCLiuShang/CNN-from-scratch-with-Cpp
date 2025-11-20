#pragma once
#include <vector>
// Tensor4D 四维张量：
// - Tensor4D 表示形状为 [N, C, H, W] 的四维张量（N:批次, C:通道, H:高, W:宽）。
// - 存储：采用行主序线性存储（std::vector<float>）。这样在遍历最内层 w（列）时内存是连续的，访问更高效。
// - 线性索引规则：((n*C + c)*H + h)*W + w，对应批次 n、通道 c、行 h、列 w。
// - 访问：提供 at(n,c,h,w) 获取/修改元素；idx(...) 用于需要时的线性下标计算。
struct Tensor4D {
    int N, C, H, W;                // 维度：批次、通道、高、宽
    std::vector<float> data;       // 线性内存存储所有元素
    Tensor4D() : N(0), C(0), H(0), W(0) {}
    Tensor4D(int N_, int C_, int H_, int W_) : N(N_), C(C_), H(H_), W(W_), data((size_t)N_*C_*H_*W_) {}
    inline int idx(int n,int c,int h,int w) const { return ((n*C + c)*H + h)*W + w; }
    float& at(int n,int c,int h,int w) { return data[idx(n,c,h,w)]; }
    const float& at(int n,int c,int h,int w) const { return data[idx(n,c,h,w)]; }
};
