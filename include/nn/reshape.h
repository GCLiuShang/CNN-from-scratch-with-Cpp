#pragma once
#include <vector>
#include "nn/tensor.h"
// 教程说明（Flatten 展平层）：
// - 作用：把空间特征张量转换成向量，便于接到全连接 Dense 层进行分类或回归。
// - 前向：按固定索引映射把 (n,c,h,w) 展平到一维下标 ((c*H + h)*W + w)。
// - 反向：将向量梯度按同样的索引规则重排回原始的 [N, C, H, W] 形状。
struct Flatten {
    int N,C,H,W, out_dim;
    std::vector<float> forward(const Tensor4D& x){ N=x.N;C=x.C;H=x.H;W=x.W; out_dim=C*H*W; std::vector<float> y((size_t)N*out_dim); for(int n=0;n<N;++n){ for(int c=0;c<C;++c) for(int h=0;h<H;++h) for(int w=0;w<W;++w){ y[(size_t)n*out_dim + ((c*H + h)*W + w)] = x.at(n,c,h,w); } } return y; }
    Tensor4D backward(const std::vector<float>& dy){ Tensor4D dx(N,C,H,W); for(int n=0;n<N;++n){ for(int c=0;c<C;++c) for(int h=0;h<H;++h) for(int w=0;w<W;++w){ dx.at(n,c,h,w) = dy[(size_t)n*out_dim + ((c*H + h)*W + w)]; } } return dx; }
};
