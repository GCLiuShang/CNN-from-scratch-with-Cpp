#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "nn/tensor.h"
#include "nn/dense.h"
#include "nn/activations.h"
#include "nn/loss.h"
#include "nn/conv2d.h"
#include "nn/pool.h"
#include "nn/reshape.h"
#include "nn/utils.h"
// 演示「从零实现」的卷积神经网络训练流程，
// 模型结构：Conv(1->16,5x5) -> ReLU -> Conv(16->32,3x3) -> ReLU -> Conv(32->64,3x3) -> ReLU ->
//           MaxPool(2x2) -> MaxPool(2x2) -> Flatten -> Dense(64*2*2 -> 2) -> SoftmaxCE
// 数据集：利用 utils.h::make_dataset 生成的合成数据（8x8 单通道），标签为 {0,1}
// 目标：分类判断图像中心是否存在 3x3 的「亮块」
// 重要概念（适合初学者）：
// - Tensor4D：形状 [N, C, H, W] 的张量；采用行主序线性内存。详见 nn/tensor.h。
// - 前向传播：逐层把输入变换为输出（特征提取 + 分类）；
// - 反向传播：根据损失对各层的参数与输入计算梯度（链式法则）；
// - 参数更新：采用最简单的 SGD（W -= lr * dW）。
// - SoftmaxCE：先做 softmax 得到概率，再用交叉熵评估分类好坏。
using namespace std;

// 训练示例入口：构建一个简单的 CNN（Conv -> ReLU -> Pool -> Flatten -> Dense），
// 在合成数据上进行分类训练与评估。整个过程包括：数据准备、前向、损失、反向、更新与评估。
int main(){
    // 1) 数据准备
    // N：批大小（一次处理的样本数）；X：输入张量 [N,1,8,8]；Y：整型标签向量（长度 N）
    int N=256; Tensor4D X; vector<int> Y; make_dataset(X,Y,N);
    // 2) 搭建网络各层（卷积核大小与通道数决定特征提取的复杂度）
    Conv2D conv1(1,16,5,5); ReLU relu1;             // 第一层：把 1 个通道映射成 16 个更丰富的特征
    Conv2D conv2(16,32,3,3); ReLU relu2;            // 第二层：继续提取更高层次的特征
    Conv2D conv3(32,64,3,3); ReLU relu3;            // 第三层：得到 64 个通道的特征图
    MaxPool2D pool1(2,2); MaxPool2D pool2(2,2);     // 两次 2x2 最大池化，下采样减小空间维度
    Flatten flat; Dense fc(64*2*2, 2); SoftmaxCE loss(2); // 展平后接全连接做二分类，配合交叉熵损失
    float lr=0.03f; int epochs=200;
    auto t0 = chrono::steady_clock::now();
    for(int ep=1; ep<=epochs; ++ep){
        // 3) 前向传播：自左至右计算各层输出
        // 说明：卷积层输出仍是 Tensor4D；ReLU 在此以向量形式处理，为了复用简单 API，需要在 Tensor4D 与向量之间来回拷贝。
        Tensor4D y1 = conv1.forward(X);
        // 将 Tensor4D 展开为一维向量，便于与 ReLU 的接口适配
        vector<float> v1(y1.data.size()); for(size_t i=0;i<v1.size();++i) v1[i]=y1.data[i];
        vector<float> v2 = relu1.forward(v1);
        // 把向量结果拷贝回 Tensor4D，继续后续卷积与池化
        Tensor4D y2(y1.N,y1.C,y1.H,y1.W); for(size_t i=0;i<v2.size();++i) y2.data[i]=v2[i];
        Tensor4D y3 = conv2.forward(y2);
        vector<float> v3(y3.data.size()); for(size_t i=0;i<v3.size();++i) v3[i]=y3.data[i];
        vector<float> v4 = relu2.forward(v3);
        Tensor4D y4(y3.N,y3.C,y3.H,y3.W); for(size_t i=0;i<v4.size();++i) y4.data[i]=v4[i];
        Tensor4D y5 = conv3.forward(y4);
        vector<float> v5(y5.data.size()); for(size_t i=0;i<v5.size();++i) v5[i]=y5.data[i];
        vector<float> v6 = relu3.forward(v5);
        Tensor4D y6(y5.N,y5.C,y5.H,y5.W); for(size_t i=0;i<v6.size();++i) y6.data[i]=v6[i];
        Tensor4D y7 = pool1.forward(y6);
        Tensor4D y8 = pool2.forward(y7);
        // 展平为 [N, C*H*W]，进入全连接层得到 logits（未归一化的类别分数）
        vector<float> vflat = flat.forward(y8);
        vector<float> logits = fc.forward(vflat, N);

        // 4) 计算损失：softmax 后对真实标签的负对数似然，取批平均
        float L = loss.forward(logits, Y, N);

        // 5) 反向传播：从损失对 logits 求梯度，逐层向前传回到输入
        vector<float> dlogits = loss.backward();
        vector<float> dfc = fc.backward(dlogits);
        Tensor4D dflat = flat.backward(dfc);
        Tensor4D dpool2 = pool2.backward(dflat);
        Tensor4D dpool1 = pool1.backward(dpool2);
        // 与前向相反，这里也需要在 Tensor4D 与向量之间做形状适配
        vector<float> vrelu3(dpool1.data.size()); for(size_t i=0;i<vrelu3.size();++i) vrelu3[i]=dpool1.data[i];
        vector<float> dact3 = relu3.backward(vrelu3);
        Tensor4D dy3(dpool1.N,dpool1.C,dpool1.H,dpool1.W); for(size_t i=0;i<dact3.size();++i) dy3.data[i]=dact3[i];
        Tensor4D dconv3 = conv3.backward(dy3);
        vector<float> vrelu2(dconv3.data.size()); for(size_t i=0;i<vrelu2.size();++i) vrelu2[i]=dconv3.data[i];
        vector<float> dact2 = relu2.backward(vrelu2);
        Tensor4D dy2(dconv3.N,dconv3.C,dconv3.H,dconv3.W); for(size_t i=0;i<dact2.size();++i) dy2.data[i]=dact2[i];
        Tensor4D dconv2 = conv2.backward(dy2);
        vector<float> vrelu1(dconv2.data.size()); for(size_t i=0;i<vrelu1.size();++i) vrelu1[i]=dconv2.data[i];
        vector<float> dact1 = relu1.backward(vrelu1);
        Tensor4D dy1(dconv2.N,dconv2.C,dconv2.H,dconv2.W); for(size_t i=0;i<dact1.size();++i) dy1.data[i]=dact1[i];
        Tensor4D dconv1 = conv1.backward(dy1);

        // 6) 参数更新：SGD 把梯度乘以学习率从参数中减去
        conv1.update(lr); conv2.update(lr); conv3.update(lr); fc.update(lr);

        if(ep%20==0){ cout<<"epoch="<<ep<<" loss="<<L<<"\n"; }
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout<<"time="<<secs<<"s\n";

    // 7) 评估：重新跑一次前向，统计预测与标签是否一致
    Tensor4D y1 = conv1.forward(X);
    vector<float> v1(y1.data.size()); for(size_t i=0;i<v1.size();++i) v1[i]=y1.data[i];
    vector<float> v2 = relu1.forward(v1);
    Tensor4D y2(y1.N,y1.C,y1.H,y1.W); for(size_t i=0;i<v2.size();++i) y2.data[i]=v2[i];
    Tensor4D y3 = conv2.forward(y2);
    vector<float> v3(y3.data.size()); for(size_t i=0;i<v3.size();++i) v3[i]=y3.data[i];
    vector<float> v4 = relu2.forward(v3);
    Tensor4D y4(y3.N,y3.C,y3.H,y3.W); for(size_t i=0;i<v4.size();++i) y4.data[i]=v4[i];
    Tensor4D y5 = conv3.forward(y4);
    vector<float> v5(y5.data.size()); for(size_t i=0;i<v5.size();++i) v5[i]=y5.data[i];
    vector<float> v6 = relu3.forward(v5);
    Tensor4D y6(y5.N,y5.C,y5.H,y5.W); for(size_t i=0;i<v6.size();++i) y6.data[i]=v6[i];
    Tensor4D y7 = pool1.forward(y6);
    Tensor4D y8 = pool2.forward(y7);
    vector<float> vflat = flat.forward(y8);
    vector<float> logits = fc.forward(vflat, N);
    // 选择每个样本的最大概率对应的类别作为预测
    int correct=0; for(int n=0;n<N;++n){ int a = max_element(logits.begin()+n*2, logits.begin()+n*2+2) - (logits.begin()+n*2); if(a==Y[n]) ++correct; }
    cout<<"acc="<< (float)correct/N <<"\n";
    return 0;
}
