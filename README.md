# CNN-from-scratch-with-Cpp

用C++从零实现的卷积神经网络，支持CUDA/SYCL加速。

## 项目简介

一个完整的CNN实现，包含：
- 卷积层 (Conv2D)
- 激活函数 (ReLU) 
- 池化层 (MaxPool)
- 全连接层 (Dense)
- Softmax交叉熵损失函数

## 特性

- 支持CUDA和SYCL GPU加速
- 适合学习深度学习原理
- 模块化设计，易于扩展
- 纯C++实现，无外部深度学习框架依赖

## 项目结构
- include/nn/ # 神经网络头文件
- activations.h # 激活函数
- conv2d.h # 卷积层
- dense.h # 全连接层
- loss.h # 损失函数
- pool.h # 池化层
- reshape.h # 展平层
- tensor.h # 张量结构
- utils.h # 工具函数
- cuda/ # CUDA内核
- sycl/ # SYCL实现
- train.cpp # 训练示例

## 构建说明

### 使用CMake构建
mkdir build && cd build
cmake ..
make

### 或者直接编译
g++ -std=c++11 train.cpp -o train
