#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        // Process() 每核共搬 tileNum * BUFFER_NUM 块，故单块长度需再除以 BUFFER_NUM
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        // tanh 用 fp32 中间量算，避免 fp16 下 e^x 抵消带来的精度损失
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // tanh(x) = (e^x - e^-x) / (e^x + e^-x)，复用 tmp0 存放 x 的 fp32 副本
        AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Exp(tmp1, tmp0, this->tileLength);                                  // tmp1 = e^x
        AscendC::Muls(tmp2, tmp0, static_cast<float>(-1.0), this->tileLength);       // tmp2 = -x
        AscendC::Exp(tmp2, tmp2, this->tileLength);                                  // tmp2 = e^-x
        AscendC::Sub(tmp0, tmp1, tmp2, this->tileLength);                            // tmp0 = e^x - e^-x
        AscendC::Add(tmp1, tmp1, tmp2, this->tileLength);                            // tmp1 = e^x + e^-x
        AscendC::Div(tmp2, tmp0, tmp1, this->tileLength);                            // tmp2 = tanh(x)
        // 收窄到 fp16 用 CAST_RINT（IEEE754 RNE），与 numpy astype 行为一致
        AscendC::Cast(yLocal, tmp2, AscendC::RoundMode::CAST_RINT, this->tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}