#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include "nn/tensor.h"
// Conv2D 二维卷积层：
// - 目标：提取局部空间特征。卷积核大小为 (kH,kW)，输出通道数为 outC（即有 outC 个不同的卷积核组）。
// - 填充：采用同尺寸（same）风格，pad = kernel/2，使输出的 H、W 与输入相同，便于堆叠多层。
// - 前向：对每个输出位置 (n,oc,h,w)，把输入相邻区域乘以对应核权重求和并加偏置 b[oc]。
//   边界按零填充（越界的输入不参与累加）。
// - 反向：
//   dW（核权重梯度）：在所有样本/空间位置上累加 x * dy；
//   db（偏置梯度）：对每个输出通道累加 dy；
//   dX（输入梯度）：将输出梯度与权重按转置卷积（反向相关）方式累加回输入位置。
// - 性能路径：若定义了宏 USE_SYCL，则调用 sycl 实现以利用 CPU/GPU 并行；否则走纯 CPU 版本。
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
extern "C" void conv2d_forward_sycl(const float* x, const float* w, const float* b, float* y,
                                     int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW);
extern "C" void conv2d_backward_dwdb_sycl(const float* x, const float* dy, float* dW, float* db,
                                           int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW);
extern "C" void conv2d_backward_dx_sycl(const float* dy, const float* w, float* dx,
                                         int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW);
sycl::queue& make_queue();
extern "C" void conv2d_forward_dev_sycl(const float* xd, const float* wd, const float* bd, float* yd,
                                         int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW);
extern "C" void conv2d_backward_dwdb_dev_sycl(const float* xd, const float* dyd, float* dWd, float* dbd,
                                               int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW);
extern "C" void conv2d_backward_dx_dev_sycl(const float* dyd, const float* wd, float* dxd,
                                             int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW);
#endif
struct Conv2D {
    int inC, outC, kH, kW, padH, padW, N_cache, H_cache, W_cache;
    std::vector<float> W, b;
    Tensor4D x_cache, y_cache;
    std::vector<float> dW, db;
    bool use_sycl;
#ifdef USE_SYCL
    sycl::queue* q;
    float *xd_dev, *wd_dev, *bd_dev, *yd_dev;
    float *dyd_dev, *dWd_dev, *dbd_dev, *dxd_dev;
    size_t xs_c, ws_w_c, ys_c, bs_c, dys_c, xs2_c, ws_dw_c;
#endif
    Conv2D(int inC_, int outC_, int kH_, int kW_) : inC(inC_), outC(outC_), kH(kH_), kW(kW_) {
        padH = kH/2; padW = kW/2;                           // 采用“same”风格的对称填充
        W.resize((size_t)outC * inC * kH * kW);
        b.resize((size_t)outC);
        std::random_device rd; std::mt19937 gen(rd()); std::normal_distribution<float> nd(0.0f, 0.1f);
        for (auto &w : W) w = nd(gen);
        for (auto &bb : b) bb = 0.0f;
        dW.assign(W.size(), 0.0f);
        db.assign(b.size(), 0.0f);
        use_sycl = false;
#ifdef USE_SYCL
        use_sycl = true;                                     // 编译时若启用 SYCL，则走加速路径
        q = nullptr; xd_dev=nullptr; wd_dev=nullptr; bd_dev=nullptr; yd_dev=nullptr; dyd_dev=nullptr; dWd_dev=nullptr; dbd_dev=nullptr; dxd_dev=nullptr;
        xs_c=0; ws_w_c=0; ys_c=0; bs_c=0; dys_c=0; xs2_c=0; ws_dw_c=0;
#endif
    }
    ~Conv2D(){
#ifdef USE_SYCL
        if(use_sycl && q){ if(xd_dev) sycl::free(xd_dev, *q); if(wd_dev) sycl::free(wd_dev, *q); if(bd_dev) sycl::free(bd_dev, *q); if(yd_dev) sycl::free(yd_dev, *q); if(dyd_dev) sycl::free(dyd_dev, *q); if(dWd_dev) sycl::free(dWd_dev, *q); if(dbd_dev) sycl::free(dbd_dev, *q); if(dxd_dev) sycl::free(dxd_dev, *q); }
#endif
    }
    Tensor4D forward(const Tensor4D& x) {
        x_cache = x; N_cache = x.N; H_cache = x.H; W_cache = x.W;
        Tensor4D y(x.N, outC, x.H, x.W);
        if(use_sycl){
#ifdef USE_SYCL
            if(!q) q = &make_queue();
            size_t xs = (size_t)x.N*inC*x.H*x.W; size_t ws = (size_t)outC*inC*kH*kW; size_t ys = (size_t)x.N*outC*x.H*x.W; size_t bs = (size_t)outC;
            conv2d_forward_sycl(x.data.data(), W.data(), b.data(), y.data.data(), x.N, inC, x.H, x.W, outC, kH, kW, padH, padW);
#else
            for (int n=0;n<x.N;++n) for (int oc=0;oc<outC;++oc) for (int h=0;h<x.H;++h) for (int w=0;w<x.W;++w) {
                float s = b[oc];
                for (int ic=0;ic<inC;++ic) for (int kh=0;kh<kH;++kh) for (int kw=0;kw<kW;++kw) {
                    int ih = h + kh - padH; int iw = w + kw - padW;
                    if (ih>=0 && ih<x.H && iw>=0 && iw<x.W) s += x.at(n,ic,ih,iw) * W[(((oc*inC)+ic)*kH + kh)*kW + kw];
                }
                y.at(n,oc,h,w) = s;
            }
#endif
        } else {
            for (int n=0;n<x.N;++n) for (int oc=0;oc<outC;++oc) for (int h=0;h<x.H;++h) for (int w=0;w<x.W;++w) {
                float s = b[oc];
                for (int ic=0;ic<inC;++ic) for (int kh=0;kh<kH;++kh) for (int kw=0;kw<kW;++kw) {
                    int ih = h + kh - padH; int iw = w + kw - padW;
                    if (ih>=0 && ih<x.H && iw>=0 && iw<x.W) s += x.at(n,ic,ih,iw) * W[(((oc*inC)+ic)*kH + kh)*kW + kw];
                }
                y.at(n,oc,h,w) = s;
            }
        }
        y_cache = y; return y;
    }
    Tensor4D backward(const Tensor4D& dy) {
        std::fill(dW.begin(), dW.end(), 0.0f); std::fill(db.begin(), db.end(), 0.0f);
        Tensor4D dx(dy.N, inC, dy.H, dy.W);
        if(use_sycl){
#ifdef USE_SYCL
            if(!q) q = &make_queue();
            size_t xs = (size_t)dy.N*inC*dy.H*dy.W; size_t dys = (size_t)dy.N*outC*dy.H*dy.W; size_t ws = (size_t)outC*inC*kH*kW; size_t bs = (size_t)outC;
            if(xs2_c!=xs){ if(dxd_dev) sycl::free(dxd_dev, *q); dxd_dev = sycl::malloc_shared<float>(xs, *q); xs2_c = xs; }
            if(dys_c!=dys){ if(dyd_dev) sycl::free(dyd_dev, *q); dyd_dev = sycl::malloc_shared<float>(dys, *q); dys_c = dys; }
            if(ws_dw_c!=ws){ if(dWd_dev) sycl::free(dWd_dev, *q); dWd_dev = sycl::malloc_shared<float>(ws, *q); ws_dw_c = ws; }
            if(bs_c!=bs){ if(dbd_dev) sycl::free(dbd_dev, *q); dbd_dev = sycl::malloc_shared<float>(bs, *q); bs_c = bs; }
            if(xs_c!= (size_t)dy.N*inC*dy.H*dy.W){ if(xd_dev) sycl::free(xd_dev, *q); xd_dev = sycl::malloc_shared<float>((size_t)dy.N*inC*dy.H*dy.W, *q); xs_c = (size_t)dy.N*inC*dy.H*dy.W; }
            if(ws_w_c!=ws){ if(wd_dev) sycl::free(wd_dev, *q); wd_dev = sycl::malloc_shared<float>(ws, *q); ws_w_c = ws; }
            std::copy(x_cache.data.begin(), x_cache.data.end(), xd_dev); std::copy(dy.data.begin(), dy.data.end(), dyd_dev); std::copy(W.begin(), W.end(), wd_dev);
            conv2d_backward_dwdb_dev_sycl(xd_dev, dyd_dev, dWd_dev, dbd_dev, dy.N, inC, dy.H, dy.W, outC, kH, kW, padH, padW);
            std::copy(dWd_dev, dWd_dev+ws, dW.begin()); std::copy(dbd_dev, dbd_dev+bs, db.begin());
            conv2d_backward_dx_dev_sycl(dyd_dev, wd_dev, dxd_dev, dy.N, inC, dy.H, dy.W, outC, kH, kW, padH, padW);
            std::copy(dxd_dev, dxd_dev+xs, dx.data.begin());
#else
            for (int n=0;n<dy.N;++n) for (int oc=0;oc<outC;++oc) for (int h=0;h<dy.H;++h) for (int w=0;w<dy.W;++w) {
                float g = dy.at(n,oc,h,w);
                db[oc] += g;
                for (int ic=0;ic<inC;++ic) for (int kh=0;kh<kH;++kh) for (int kw=0;kw<kW;++kw) {
                    int ih = h + kh - padH; int iw = w + kw - padW;
                    if (ih>=0 && ih<dy.H && iw>=0 && iw<dy.W) dW[(((oc*inC)+ic)*kH + kh)*kW + kw] += x_cache.at(n,ic,ih,iw) * g;
                }
            }
            for (int n=0;n<dy.N;++n) for (int ic=0;ic<inC;++ic) for (int h=0;h<dy.H;++h) for (int w=0;w<dy.W;++w) {
                float s = 0.0f;
                for (int oc=0;oc<outC;++oc) for (int kh=0;kh<kH;++kh) for (int kw=0;kw<kW;++kw) {
                    int oh = h - kh + padH; int ow = w - kw + padW;
                    if (oh>=0 && oh<dy.H && ow>=0 && ow<dy.W) s += dy.at(n,oc,oh,ow) * W[(((oc*inC)+ic)*kH + kh)*kW + kw];
                }
                dx.at(n,ic,h,w) = s;
            }
#endif
        } else {
            for (int n=0;n<dy.N;++n) for (int oc=0;oc<outC;++oc) for (int h=0;h<dy.H;++h) for (int w=0;w<dy.W;++w) {
                float g = dy.at(n,oc,h,w);
                db[oc] += g;
                for (int ic=0;ic<inC;++ic) for (int kh=0;kh<kH;++kh) for (int kw=0;kw<kW;++kw) {
                    int ih = h + kh - padH; int iw = w + kw - padW;
                    if (ih>=0 && ih<dy.H && iw>=0 && iw<dy.W) dW[(((oc*inC)+ic)*kH + kh)*kW + kw] += x_cache.at(n,ic,ih,iw) * g;
                }
            }
            for (int n=0;n<dy.N;++n) for (int ic=0;ic<inC;++ic) for (int h=0;h<dy.H;++h) for (int w=0;w<dy.W;++w) {
                float s = 0.0f;
                for (int oc=0;oc<outC;++oc) for (int kh=0;kh<kH;++kh) for (int kw=0;kw<kW;++kw) {
                    int oh = h - kh + padH; int ow = w - kw + padW;
                    if (oh>=0 && oh<dy.H && ow>=0 && ow<dy.W) s += dy.at(n,oc,oh,ow) * W[(((oc*inC)+ic)*kH + kh)*kW + kw];
                }
                dx.at(n,ic,h,w) = s;
            }
        }
        return dx;
    }
    void update(float lr) {               // SGD 参数更新
        for (size_t i=0;i<W.size();++i) W[i] -= lr * dW[i];
        for (size_t i=0;i<b.size();++i) b[i] -= lr * db[i];
    }
};
