# TanhCustom

基于昇腾 Ascend C 的自定义 `Tanh` 算子实现，采用 **Tiling 模板编程**：host 侧下发 tiling 参数，kernel 侧用纯 POD 结构体接收。

已在 **Ascend 910B + CANN 9.0.0** 上完成端到端验证，精度校验通过。

---

## 算子定义

```
tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

| 项 | 值 |
|---|---|
| 输入 `x` | `float16`，shape `(8, 2048)` |
| 输出 `y` | `float16`，shape `(8, 2048)` |
| 中间计算 | `float32`（见「实现要点」） |
| 精度判据 | 相对误差 `1e-3`，最小比较阈值 `1e-3` |

---

## 目录结构

```
TanhCustom/                        # 仓库根
├── TanhCustom/                    # 考试包原目录，故有一层同名嵌套
│   ├── AclNNInvocation/           # aclnn 调用样例（脚手架自带，未改）
│   │   ├── run.sh                 # 生成数据 → 编译 → 执行 → 比对
│   │   └── scripts/gen_data.py    # 生成 input_x.bin 与 golden.bin
│   └── TanhCustom/                # 算子工程：build.sh 从这里构建出 .run 安装包
│       ├── CMakePresets.json      # ★ CANN 安装路径在这里配置
│       ├── build.sh
│       ├── framework/tf_plugin/   # TensorFlow 插件（脚手架自带，未改）
│       ├── op_host/
│       │   └── tanh_custom.cpp    # TilingFunc / InferShape / OpDef
│       └── op_kernel/
│           ├── tanh_custom.cpp    # KernelTanh：CopyIn → Compute → CopyOut
│           └── tanh_custom_tiling.h  # Tiling 数据结构（host / kernel 共用）
├── LICENSE
└── local_test/
    └── simulate_tanh.py           # 离线自测：无 NPU、无 CANN 也能跑
```

---

## 环境要求

- 昇腾 NPU（本工程在 **910B** 上验证）
- **CANN 9.0.0**
- Python 3 + `numpy`
- `cmake` ≥ 3.16

> GitCode Notebook 默认安装的是 CANN **8.5.0**，需要先升级到 9.0.0。

### 必改：CMakePresets.json 里的 CANN 路径

`TanhCustom/TanhCustom/CMakePresets.json` 中的 `ASCEND_CANN_PACKAGE_PATH` 必须指向**你本机 CANN 9.0.0 的实际安装位置**。`build.sh` 走的是 `cmake --preset=default`，这个值是唯一的配置入口，不会在命令行被覆盖。

```json
"ASCEND_CANN_PACKAGE_PATH": { "type": "PATH", "value": "/path/to/your/cann-9.0.0" }
```

---

## 构建 / 安装 / 测试

**① 整套 source 环境，不要逐变量覆盖**

```bash
source <你的CANN路径>/set_env.sh
```

**② 构建算子包**

```bash
cd TanhCustom/TanhCustom
bash build.sh
```

产物：`build_out/custom_opp_ubuntu_aarch64.run`

**③ 安装算子包**

```bash
cd build_out
./custom_opp_ubuntu_aarch64.run
```

不带参数即运行内嵌安装脚本，成功后打印 `SUCCESS`。

**④ 跑精度测试**

```bash
cd ../../AclNNInvocation
ASCEND_HOME_DIR=<你的CANN路径> bash run.sh
```

看到下面这段即为通过：

```
INFO: you have passed the Precision!
```

---

## 实现要点

### Tiling 分块

| 常量 | 值 | 含义 |
|---|---|---|
| `BLOCK_DIM` | 8 | 使用的核数 |
| `TILE_NUM` | 8 | 每核切分块数 |
| `BUFFER_NUM` | 2 | 双缓冲队列深度 |

单核处理 `totalLength / BLOCK_DIM` 个元素，每核循环 `TILE_NUM * BUFFER_NUM` 次，因此单块长度为：

```
tileLength = blockLength / TILE_NUM / BUFFER_NUM
```

### 用 fp32 算中间量

`float16` 在 `x ≈ 1` 附近的分辨率约 `2^-10 ≈ 9.77e-4`。若直接在半精度下计算 `e^x - e^-x`，会把两个已被舍入成同一个数的量相减，**小值被直接抹成 0**；而这种绝对误差本身很小，`1e-3` 的容差判据几乎抓不住它。

因此 kernel 先把输入 `Cast` 到 `float32`，全程用 fp32 跑 `Exp / Muls / Sub / Add / Div`，最后再收窄回 `float16`：

```cpp
AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength); // fp16 -> fp32
AscendC::Exp(tmp1, tmp0, this->tileLength);                                   // e^x
AscendC::Muls(tmp2, tmp0, static_cast<float>(-1.0), this->tileLength);
AscendC::Exp(tmp2, tmp2, this->tileLength);                                   // e^-x
AscendC::Sub(tmp0, tmp1, tmp2, this->tileLength);                             // e^x - e^-x
AscendC::Add(tmp1, tmp1, tmp2, this->tileLength);                             // e^x + e^-x
AscendC::Div(tmp2, tmp0, tmp1, this->tileLength);                             // tanh(x)
AscendC::Cast(yLocal, tmp2, AscendC::RoundMode::CAST_RINT, this->tileLength); // fp32 -> fp16
```

收窄用 `CAST_RINT`（IEEE754 round-to-nearest-even），与 `numpy.astype(np.float16)` 行为一致。

### Tiling 数据传递

`tanh_custom_tiling.h` 中是一个**普通 POD 结构体**，不是 `BEGIN_TILING_DATA_DEF` 宏那一套：

```cpp
struct TanhCustomTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};
```

因此 host 侧直接取指针写字段即可，**不需要 `SaveToBuffer` / `memcpy`**：

```cpp
TanhCustomTilingData* tiling = context->GetTilingData<TanhCustomTilingData>();
tiling->totalLength = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());
tiling->tileNum = TILE_NUM;
context->SetBlockDim(BLOCK_DIM);
```

---

## 移植注意事项

以下是实际踩过的坑，换机器时按顺序检查。

### 1. `GetBlockIdx` / `GetBlockNum` 需要 `AscendC::` 限定

CANN 9.0.0 起这两个函数位于 `AscendC` 命名空间内，不加限定会直接编译失败（`GetBlockIdx` 在此声明）：

```cpp
this->blockLength = totalLength / AscendC::GetBlockNum();
xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
```

### 2. 升级 CANN 要整套 `source`，别只改一个变量

多版本共存的机器上（例如 `/usr/local/Ascend/cann-8.5.0` 与自装的 9.0.0 同时存在），`ASCEND_HOME_PATH` / `CMAKE_PREFIX_PATH` / `LD_LIBRARY_PATH` / `ASCEND_OPP_PATH` 是**相互独立**的。只覆盖其中一个，编译期与运行期就会来自不同版本，症状是 `undefined symbol` 或 `include could not find requested file` —— 看着像代码写错了。

正确做法：整套 `source <CANN>/set_env.sh`，换完再 `env | grep -i ascend` 确认没有旧版本残留。

### 3. `.run` 安装包不要加 `--install`

`./custom_opp_ubuntu_aarch64.run` 不带参数就是运行内嵌的安装脚本。`--install` 不是这个 makeself 包的有效选项，加了只会打印用法说明。

### 4. `run.sh` 可能选中错误的 CANN

`AclNNInvocation/run.sh` 中 `ASCEND_HOME_DIR` 的回退顺序是 `$HOME/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/ascend-toolkit/latest`。若 CANN 9.0.0 不在这两处，须显式指定：

```bash
ASCEND_HOME_DIR=/your/cann-9.0.0 bash run.sh
```

否则会 source 到旧版本的 `setenv.bash`，编译出来的 acl 可执行文件链接的是另一套库。

### 5. 行尾必须是 LF

仓库根的 `.gitattributes` 已锁定 `* text=auto eol=lf`。从 Windows 打包或传输时注意确认 —— shell 脚本带 CRLF 在 Linux 上会报 `bad interpreter: /bin/bash^M`。

---

## 离线自测

`local_test/simulate_tanh.py` 不需要 NPU，也不需要 CANN，Windows 上可直接运行：

```bash
python local_test/simulate_tanh.py
```

它复刻了 host 侧的分块逻辑与 kernel 侧的算子实现，并采用与 `verify_result.py` 完全一致的容差判据判定。

注意：这**不是** NPU 的位精确仿真，`Exp` / `Div` 的末位与 numpy 不会逐位相同。它的价值在于提前验证**分块逻辑正确**与**精度余量足够**这两件事。

---

## 许可证

本工程基于华为昇腾 Ascend C 算子开发脚手架修改而来。仓库内文件按来源分属不同条款，**不是单一许可证**：

| 文件 | 版权 | 条款 |
|---|---|---|
| `op_host/tanh_custom.cpp`、`op_kernel/tanh_custom.cpp` | 本仓库 | **Mulan PSL v2** —— 全文见根目录 [`LICENSE`](LICENSE) |
| `op_kernel/tanh_custom_tiling.h`、`framework/tf_plugin/tensorflow_tanh_custom_plugin.cc` | 华为（MindStudio 模板） | Mulan PSL v2，文件头已声明 |
| 各层 `CMakeLists.txt` | 华为 | CANN Open Software License Agreement Version 2.0 |
| `AclNNInvocation/`（脚手架自带，未作修改） | 华为 | 文件头仅写 "All rights reserved"，未附授权条款 |

脚手架文件头里引用的 `LICENSE` 指的是 **CANN OSL 全文**（随 CANN 工具链分发，本仓库未附带，故该引用悬空）。
根目录这份 `LICENSE` 是本仓库自己新增的 **Mulan PSL v2**，只覆盖上表第一行那两份源文件。
