#pragma once
#include <vector>
#include <algorithm>
#include "nn/tensor.h"
// MaxPool2D 最大池化层：
// - 作用：下采样（减小 H/W），保留每个局部窗口中的最大响应，增强平移不变性、降低参数/计算量。
// - 参数：pool 为窗口大小（正方形 pool×pool），stride 为步长；通常取 stride = pool 做非重叠池化。
// - 前向：对每个输出位置 (oh,ow)，在输入对应的窗口内找到最大值及其线性索引，写入输出并缓存索引到 idxs。
// - 反向：梯度仅回传到「前向最大值的位置」，其它位置梯度为 0；因此需要依赖前向保存的 idxs 路由梯度。
struct MaxPool2D {
    int pool, stride; int N,C,H,W, outH, outW; std::vector<int> idxs; // idxs 保存输入的线性索引
    MaxPool2D(int pool_=2,int stride_=2):pool(pool_),stride(stride_){}
    Tensor4D forward(const Tensor4D& x){
        N=x.N;C=x.C;H=x.H;W=x.W; outH=H/stride; outW=W/stride; Tensor4D y(N,C,outH,outW); idxs.assign((size_t)N*C*outH*outW, -1);
        for(int n=0;n<N;++n) for(int c=0;c<C;++c) for(int oh=0;oh<outH;++oh) for(int ow=0;ow<outW;++ow){
            float m=-1e9f; int mi=-1; int h0=oh*stride, w0=ow*stride;
            for(int kh=0;kh<pool;++kh) for(int kw=0;kw<pool;++kw){
                int ih=h0+kh, iw=w0+kw; float v=x.at(n,c,ih,iw); int id=(((n*C+c)*H)+ih)*W+iw;
                if(v>m){m=v;mi=id;}
            }
            y.at(n,c,oh,ow)=m; idxs[((n*C+c)*outH+oh)*outW+ow]=mi;
        }
        return y;
    }
    Tensor4D backward(const Tensor4D& dy){
        Tensor4D dx(N,C,H,W);
        std::fill(dx.data.begin(), dx.data.end(), 0.0f);
        for(int n=0;n<N;++n) for(int c=0;c<C;++c) for(int oh=0;oh<outH;++oh) for(int ow=0;ow<outW;++ow){
            int id=idxs[((n*C+c)*outH+oh)*outW+ow]; int ih=(id%(H*W))/W; int iw=id%W; dx.at(n,c,ih,iw)+=dy.at(n,c,oh,ow);
        }
        return dx;
    }
};
