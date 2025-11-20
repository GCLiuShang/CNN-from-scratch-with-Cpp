#pragma once
#include <vector>
#include <random>
#include "nn/tensor.h"
// Utils 数据与随机工具：
// - RNG：封装了均匀分布与整数随机的调用，便于生成合成数据。
// - make_dataset：构造一个简单的二分类数据集：
//   类 1：在 8x8 图像中心（[2..4]×[2..4]）放置一个 3x3 的亮块（值=1），再叠加微小噪声；
//   类 0：全零图像，再叠加微小噪声；
//   这样网络需要学习到“中心亮块”的判别性局部特征。
struct RNG { std::mt19937 gen; RNG(){ std::random_device rd; gen.seed(rd()); } float uniform(float a,float b){ std::uniform_real_distribution<float> d(a,b); return d(gen);} int randint(int a,int b){ std::uniform_int_distribution<int> d(a,b-1); return d(gen);} };
inline void make_dataset(Tensor4D& X, std::vector<int>& Y, int N){
    X = Tensor4D(N,1,8,8); Y.resize(N); RNG rng;
    for(int n=0;n<N;++n){ int label = rng.randint(0,2); Y[n]=label; for(int h=0;h<8;++h) for(int w=0;w<8;++w){ X.at(n,0,h,w)=0.0f; }
        if(label==1){ for(int h=2;h<=4;++h) for(int w=2;w<=4;++w) X.at(n,0,h,w)=1.0f; }
        for(int h=0;h<8;++h) for(int w=0;w<8;++w){ X.at(n,0,h,w) += rng.uniform(-0.05f,0.05f); }
    }
}
