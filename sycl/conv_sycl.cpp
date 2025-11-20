#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
// Conv2D SYCL 加速版卷积与反向：
// - 目的：利用 SYCL 在 CPU/GPU 上并行计算，提升前向与反向的吞吐。
// - 设备选择：若环境变量 SYCL_DEVICE_FILTER 包含 "cpu"，则优先选 CPU；否则尝试 GPU，失败则回退到 CPU。
// - 内存：使用 malloc_device 分配显存、copy 进行主设备间拷贝；部分 dev 版本使用 shared 内存减少显式拷贝。
// - 并行策略：将输出/权重/输入的一维总元素数作为全局范围，设置合适的局部工作组大小（如 128/256），
//   通过一维索引映射回 (n, oc, h, w) 或权重下标，内核中完成对应的累加计算。
// - 索引映射示例（前向）：
//   idx -> n = idx/(outC*H*W);
//          oc = (idx%(outC*H*W))/(H*W);
//          h  = ((idx%(H*W))/W);
//          w  = (idx%W);
// - SYCL 提交的 kernel 是异步执行的；在需要取回结果时使用 .wait() 或依赖 copy 的隐式同步。
using namespace sycl;

// 统一构建并缓存一个队列：避免每次调用重复创建设备队列带来的开销
queue& make_queue(){
    static bool printed = false;
    static queue* qp = nullptr;
    if(!qp){
        const char* f = std::getenv("SYCL_DEVICE_FILTER");
        if (f) {
            std::string s(f);
            if (s.find("cpu") != std::string::npos) { qp = new queue(cpu_selector_v, {property::queue::in_order{}}); if(!printed){ std::cout << "SYCL device: " << qp->get_device().get_info<info::device::name>() << std::endl; printed=true; } return *qp; }
        }
        try { qp = new queue(gpu_selector_v, {property::queue::in_order{}}); } catch(...) { qp = new queue(cpu_selector_v, {property::queue::in_order{}}); }
        if(!printed){ std::cout << "SYCL device: " << qp->get_device().get_info<info::device::name>() << std::endl; printed=true; }
    }
    return *qp;
}

// SYCL 前向卷积：将各位置映射为一维索引并在内核中计算加权和（边界零填充）
extern "C" void conv2d_forward_sycl(const float* x, const float* w, const float* b, float* y,
                                     int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    queue q = make_queue();
    size_t xs = (size_t)N*inC*H*W, ws = (size_t)outC*inC*kH*kW, ys = (size_t)N*outC*H*W, bs = (size_t)outC;
    float* xd = malloc_device<float>(xs, q); float* wd = malloc_device<float>(ws, q); float* yd = malloc_device<float>(ys, q); float* bd = malloc_device<float>(bs, q);
    q.copy(x, xd, xs); q.copy(w, wd, ws); q.copy(b, bd, bs);
    size_t total = ys; size_t local = 256; size_t global = ((total+local-1)/local)*local; // 向上取整保证整除
    q.parallel_for(nd_range<1>(range<1>(global), range<1>(local)), [=](nd_item<1> it){
        size_t idx = it.get_global_id(0); if(idx>=total) return; int n = idx/(outC*H*W); int rem1 = idx%(outC*H*W); int oc = rem1/(H*W); int rem2 = rem1%(H*W); int h = rem2/W; int ww = rem2%W; float s = bd[oc];
        for(int ic=0; ic<inC; ++ic) for(int kh=0; kh<kH; ++kh) for(int kw=0; kw<kW; ++kw){ int ih = h + kh - padH; int iw = ww + kw - padW; if(ih>=0 && ih<H && iw>=0 && iw<W){ size_t xi = (((n*inC+ic)*H)+ih)*W+iw; size_t wi = (((oc*inC+ic)*kH)+kh)*kW+kw; s += xd[xi]*wd[wi]; }}
        yd[idx] = s;
    });
    q.copy(yd, y, ys).wait();
    free(xd, q); free(wd, q); free(yd, q); free(bd, q);
}

// SYCL 反向计算 dW/db：对每个核元素累加输入与输出梯度贡献（权重梯度与偏置梯度）
extern "C" void conv2d_backward_dwdb_sycl(const float* x, const float* dy, float* dW, float* db,
                                           int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    queue& q = make_queue();
    size_t xs = (size_t)N*inC*H*W, dys = (size_t)N*outC*H*W, ws = (size_t)outC*inC*kH*kW, bs = (size_t)outC;
    static float* xd = nullptr; static size_t xs_c = 0;
    static float* dyd = nullptr; static size_t dys_c = 0;
    static float* dWd = nullptr; static size_t ws_c = 0;
    static float* dbd = nullptr; static size_t bs_c = 0;
    if(xs!=xs_c){ if(xd) free(xd, q); xd = malloc_device<float>(xs, q); xs_c = xs; } // 尺寸变化时重新分配设备内存
    if(dys!=dys_c){ if(dyd) free(dyd, q); dyd = malloc_device<float>(dys, q); dys_c = dys; }
    if(ws!=ws_c){ if(dWd) free(dWd, q); dWd = malloc_device<float>(ws, q); ws_c = ws; }
    if(bs!=bs_c){ if(dbd) free(dbd, q); dbd = malloc_device<float>(bs, q); bs_c = bs; }
    q.copy(x, xd, xs); q.copy(dy, dyd, dys);
    float* xd_ptr = xd; float* dyd_ptr = dyd; float* dWd_ptr = dWd; float* dbd_ptr = dbd;
    size_t total_w = ws; size_t local = 128; size_t global_w = ((total_w+local-1)/local)*local; // 逐权重元素并行
    q.parallel_for(nd_range<1>(range<1>(global_w), range<1>(local)), [=](nd_item<1> it){
        size_t wi_idx = it.get_global_id(0); if(wi_idx>=total_w) return; int tmp=wi_idx; int kw = tmp%kW; tmp/=kW; int kh = tmp%kH; tmp/=kH; int ic = tmp%inC; int oc = tmp/inC; float acc = 0.0f;
        for(int n=0;n<N;++n) for(int h=0;h<H;++h) for(int w_=0;w_<W;++w_){ int ih=h+kh-padH; int iw=w_+kw-padW; if(ih>=0 && ih<H && iw>=0 && iw<W){ size_t xi=(((n*inC+ic)*H)+ih)*W+iw; size_t dyi=(((n*outC+oc)*H)+h)*W+w_; acc += xd_ptr[xi]*dyd_ptr[dyi]; }}
        dWd_ptr[wi_idx] = acc;
    });
    size_t total_b = bs; size_t global_b = ((total_b+local-1)/local)*local;
    q.parallel_for(nd_range<1>(range<1>(global_b), range<1>(local)), [=](nd_item<1> it){
        size_t oc = it.get_global_id(0); if(oc>=total_b) return; float bacc=0.0f; for(int n=0;n<N;++n) for(int h=0;h<H;++h) for(int w_=0;w_<W;++w_) bacc += dyd_ptr[(((n*outC+oc)*H)+h)*W+w_]; dbd_ptr[oc]=bacc;
    });
    q.copy(dWd, dW, ws); q.copy(dbd, db, bs).wait();
}

// SYCL 反向计算 dX：以“转置卷积/相关”方式把输出梯度与核权重累加到输入梯度
extern "C" void conv2d_backward_dx_sycl(const float* dy, const float* w, float* dx,
                                         int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    queue& q = make_queue();
    size_t dys = (size_t)N*outC*H*W, ws = (size_t)outC*inC*kH*kW, xs = (size_t)N*inC*H*W;
    static float* dyd = nullptr; static size_t dys_c = 0;
    static float* wd = nullptr; static size_t ws_c = 0;
    static float* dxd = nullptr; static size_t xs_c = 0;
    if(dys!=dys_c){ if(dyd) free(dyd, q); dyd = malloc_device<float>(dys, q); dys_c = dys; }
    if(ws!=ws_c){ if(wd) free(wd, q); wd = malloc_device<float>(ws, q); ws_c = ws; }
    if(xs!=xs_c){ if(dxd) free(dxd, q); dxd = malloc_device<float>(xs, q); xs_c = xs; }
    q.copy(dy, dyd, dys); q.copy(w, wd, ws);
    float* dyd_ptr = dyd; float* wd_ptr = wd; float* dxd_ptr = dxd;
    size_t total_x = xs; size_t local = 128; size_t global_x = ((total_x+local-1)/local)*local; // 逐输入元素并行
    q.parallel_for(nd_range<1>(range<1>(global_x), range<1>(local)), [=](nd_item<1> it){
        size_t xi_idx = it.get_global_id(0); if(xi_idx>=total_x) return; int tmp=xi_idx; int w_=tmp%W; tmp/=W; int h=tmp%H; tmp/=H; int ic=tmp%inC; int n=tmp/inC; float s=0.0f;
        for(int oc=0; oc<outC; ++oc) for(int kh=0; kh<kH; ++kh) for(int kw=0; kw<kW; ++kw){ int oh=h-kh+padH; int ow=w_-kw+padW; if(oh>=0 && oh<H && ow>=0 && ow<W){ size_t dyi=(((n*outC+oc)*H)+oh)*W+ow; size_t wi=(((oc*inC+ic)*kH)+kh)*kW+kw; s += dyd_ptr[dyi]*wd_ptr[wi]; }}
        dxd_ptr[xi_idx] = s;
    });
    q.copy(dxd, dx, xs).wait();
}
extern "C" void conv2d_forward_dev_sycl(const float* xd, const float* wd, const float* bd, float* yd,
                                         int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    queue& q = make_queue();
    size_t ys = (size_t)N*outC*H*W;
    size_t total = ys; size_t local = 256; size_t global = ((total+local-1)/local)*local;
    q.parallel_for(nd_range<1>(range<1>(global), range<1>(local)), [=](nd_item<1> it){
        size_t idx = it.get_global_id(0); if(idx>=total) return; int n = (int)(idx/(outC*H*W)); size_t rem1 = idx%(outC*H*W); int oc = (int)(rem1/(H*W)); size_t rem2 = rem1%(H*W); int h = (int)(rem2/W); int ww = (int)(rem2%W); float s = bd[oc];
        for(int ic=0; ic<inC; ++ic) for(int kh=0; kh<kH; ++kh) for(int kw=0; kw<kW; ++kw){ int ih = h + kh - padH; int iw = ww + kw - padW; if(ih>=0 && ih<H && iw>=0 && iw<W){ size_t xi = (((size_t)n*inC+ic)*H + ih)*W + iw; size_t wi = (((size_t)oc*inC+ic)*kH + kh)*kW + kw; s += xd[xi]*wd[wi]; }}
        yd[idx] = s;
    });
}
extern "C" void conv2d_backward_dwdb_dev_sycl(const float* xd, const float* dyd, float* dWd, float* dbd,
                                               int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    queue& q = make_queue();
    size_t ws = (size_t)outC*inC*kH*kW, bs = (size_t)outC;
    size_t total_w = ws; size_t local = 128; size_t global_w = ((total_w+local-1)/local)*local;
    q.parallel_for(nd_range<1>(range<1>(global_w), range<1>(local)), [=](nd_item<1> it){
        size_t wi_idx = it.get_global_id(0); if(wi_idx>=total_w) return; size_t tmp=wi_idx; int kw = (int)(tmp%kW); tmp/=kW; int kh = (int)(tmp%kH); tmp/=kH; int ic = (int)(tmp%inC); int oc = (int)(tmp/inC); float acc = 0.0f;
        for(int n=0;n<N;++n) for(int h=0;h<H;++h) for(int w_=0;w_<W;++w_){ int ih=h+kh-padH; int iw=w_+kw-padW; if(ih>=0 && ih<H && iw>=0 && iw<W){ size_t xi=(((size_t)n*inC+ic)*H + ih)*W + iw; size_t dyi=(((size_t)n*outC+oc)*H + h)*W + w_; acc += xd[xi]*dyd[dyi]; }}
        dWd[wi_idx] = acc;
    });
    size_t total_b = bs; size_t global_b = ((total_b+local-1)/local)*local;
    q.parallel_for(nd_range<1>(range<1>(global_b), range<1>(local)), [=](nd_item<1> it){
        size_t oc = it.get_global_id(0); if(oc>=total_b) return; float bacc=0.0f; for(int n=0;n<N;++n) for(int h=0;h<H;++h) for(int w_=0;w_<W;++w_) bacc += dyd[(((size_t)n*outC+oc)*H + h)*W + w_]; dbd[oc]=bacc;
    });
}
extern "C" void conv2d_backward_dx_dev_sycl(const float* dyd, const float* wd, float* dxd,
                                             int N,int inC,int H,int W,int outC,int kH,int kW,int padH,int padW){
    queue& q = make_queue();
    size_t xs = (size_t)N*inC*H*W;
    size_t total_x = xs; size_t local = 128; size_t global_x = ((total_x+local-1)/local)*local;
    q.parallel_for(nd_range<1>(range<1>(global_x), range<1>(local)), [=](nd_item<1> it){
        size_t xi_idx = it.get_global_id(0); if(xi_idx>=total_x) return; size_t tmp=xi_idx; int w_=(int)(tmp%W); tmp/=W; int h=(int)(tmp%H); tmp/=H; int ic=(int)(tmp%inC); int n=(int)(tmp/inC); float s=0.0f;
        for(int oc=0; oc<outC; ++oc) for(int kh=0; kh<kH; ++kh) for(int kw=0; kw<kW; ++kw){ int oh=h-kh+padH; int ow=w_-kw+padW; if(oh>=0 && oh<H && ow>=0 && ow<W){ size_t dyi=(((size_t)n*outC+oc)*H + oh)*W + ow; size_t wi=(((size_t)oc*inC+ic)*kH + kh)*kW + kw; s += dyd[dyi]*wd[wi]; }}
        dxd[xi_idx] = s;
    });
}
