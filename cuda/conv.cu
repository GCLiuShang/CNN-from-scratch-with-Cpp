#include <cuda_runtime.h>
extern "C" {
// CUDA 前向卷积核：每个线程计算 (n, oc, h, w) 的输出值
__global__ void conv2d_forward(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y,
                               int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    int n = blockIdx.z; int oc = blockIdx.y*blockDim.y + threadIdx.y; int hw = blockIdx.x*blockDim.x + threadIdx.x; if(oc>=outC || hw>=H*W) return; int h=hw/W, ww=hw%W; float s=b[oc];
    for(int ic=0; ic<inC; ++ic){ for(int kh=0; kh<kH; ++kh){ for(int kw=0; kw<kW; ++kw){ int ih=h+kh-padH; int iw=ww+kw-padW; if(ih>=0 && ih<H && iw>=0 && iw<W){ int xi=(((n*inC+ic)*H)+ih)*W+iw; int wi=(((oc*inC+ic)*kH)+kh)*kW+kw; s += x[xi]*w[wi]; } } } }
    y[(((n*outC+oc)*H)+h)*W+ww] = s;
}

// CUDA 反向核（dW/db）：遍历核元素并累加梯度贡献
__global__ void conv2d_backward_dW_db(const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ dW, float* __restrict__ db,
                                      int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    int oc = blockIdx.y*blockDim.y + threadIdx.y; int ic = blockIdx.x*blockDim.x + threadIdx.x; if(oc>=outC || ic>=inC) return;
    for(int kh=0; kh<kH; ++kh){ for(int kw=0; kw<kW; ++kw){ float acc=0.0f; for(int n=0;n<N;++n){ for(int h=0;h<H;++h){ for(int w_=0;w_<W;++w_){ int ih=h+kh-padH; int iw=w_+kw-padW; if(ih>=0 && ih<H && iw>=0 && iw<W){ int xi=(((n*inC+ic)*H)+ih)*W+iw; int dyi=(((n*outC+oc)*H)+h)*W+w_; acc += x[xi]*dy[dyi]; } } } }
            dW[(((oc*inC+ic)*kH)+kh)*kW+kw] = acc; }
    }
    float bacc=0.0f; for(int n=0;n<N;++n){ for(int h=0;h<H;++h){ for(int w_=0;w_<W;++w_){ bacc += dy[(((n*outC+oc)*H)+h)*W+w_]; } } } db[oc]=bacc;
}

// CUDA 反向核（dX）：用输出梯度与卷积核累加到输入梯度
__global__ void conv2d_backward_dx(const float* __restrict__ dy, const float* __restrict__ w, float* __restrict__ dx,
                                   int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    int n=blockIdx.z; int ic=blockIdx.y*blockDim.y + threadIdx.y; int hw=blockIdx.x*blockDim.x + threadIdx.x; if(ic>=inC || hw>=H*W) return; int h=hw/W, ww=hw%W; float s=0.0f;
    for(int oc=0; oc<outC; ++oc){ for(int kh=0; kh<kH; ++kh){ for(int kw=0; kw<kW; ++kw){ int oh=h-kh+padH; int ow=ww-kw+padW; if(oh>=0 && oh<H && ow>=0 && ow<W){ int dyi=(((n*outC+oc)*H)+oh)*W+ow; int wi=(((oc*inC+ic)*kH)+kh)*kW+kw; s += dy[dyi]*w[wi]; } } } }
    dx[(((n*inC+ic)*H)+h)*W+ww] = s;
}

// 前向 launch 封装：配置网格并发起 kernel
void conv2d_forward_launch(const float* x, const float* w, const float* b, float* y,
                           int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    dim3 block(256,1,1);
    dim3 grid((H*W+255)/256, outC, N);
    conv2d_forward<<<grid, block>>>(x,w,b,y,N,inC,H,W,outC,kH,kW,padH,padW);
}

// dW/db launch 封装
void conv2d_backward_dW_db_launch(const float* x, const float* dy, float* dW, float* db,
                                  int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    dim3 block(32,32,1);
    dim3 grid((inC+31)/32, (outC+31)/32, 1);
    conv2d_backward_dW_db<<<grid, block>>>(x,dy,dW,db,N,inC,H,W,outC,kH,kW,padH,padW);
}

// dX launch 封装
void conv2d_backward_dx_launch(const float* dy, const float* w, float* dx,
                               int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    dim3 block(256,1,1);
    dim3 grid((H*W+255)/256, inC, N);
    conv2d_backward_dx<<<grid, block>>>(dy,w,dx,N,inC,H,W,outC,kH,kW,padH,padW);
}

}