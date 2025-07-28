# QXDM Log Analysis Pipeline

优化的QXDM日志复杂性分析管道，支持Windows本地和Linux HPC集群环境。

## 🚀 主要特性

- **自适应性能优化**: 根据系统资源自动调整处理参数
- **多平台支持**: Windows (批处理模式) 和 Linux (多进程模式)
- **GPU加速**: 支持CUDA、MPS(Apple Silicon)和CPU模式
- **流畅进度显示**: 详细的进度条和时间估算
- **断点续传**: 支持句子提取和嵌入计算的检查点
- **可扩展性**: 支持大规模QXDM日志文件处理

## 📁 目录结构

```
3gpp_Complexity/
├── src/
│   ├── tspec_metrics_2.py      # 主分析脚本 (已优化)
│   └── metric_utils.py         # 指标计算工具
├── scripts/
│   └── tspec.sh               # SLURM HPC作业脚本
├── test_logs/                 # 测试日志文件
│   ├── highway_test.txt
│   ├── urban_analysis.log
│   └── rural_coverage.txt
├── QXDM_Logs/                 # 默认QXDM日志目录
├── test_qxdm.bat             # Windows测试脚本
└── README_QXDM.md            # 本文档
```

## 🛠️ 环境配置

### Windows 本地环境

```powershell
# 激活虚拟环境
myenv\Scripts\activate

# 设置QXDM日志路径 (可选，默认为 ./QXDM_Logs)
$env:QXDM_ROOT = "C:\path\to\your\qxdm\logs"

# 运行分析
python src\tspec_metrics_2.py
```

### Linux HPC 集群

```bash
# 提交SLURM作业
sbatch scripts/tspec.sh

# 或设置环境变量后直接运行
export QXDM_ROOT=/path/to/qxdm/logs
export BLOCK_SIZE=2000000  # 大内存系统建议值
export BATCH_SIZE=1024     # GPU批处理大小
python src/tspec_metrics_2.py --wandb-project QXDM-Analysis
```

## 🔧 性能优化配置

代码会自动检测系统配置并优化参数：

### Windows 系统 (您的配置)
- **CPU**: 16线程 → 使用6个处理线程 (避免spaCy多进程问题)
- **内存**: 64GB → 块大小500,000字符
- **GPU**: RTX 2080 8GB → 批处理大小512 (自动调整为384以适应长句子)
- **处理模式**: 批处理模式 (单进程，避免Windows多进程挂起)

### Linux HPC 系统
- **CPU**: 32核心 → 使用最多8个进程
- **内存**: 160GB → 块大小2,000,000字符  
- **GPU**: A100/V100 → 批处理大小1024
- **处理模式**: 多进程模式 (充分利用多核心)

## 📊 输出文件

分析完成后会生成以下文件：

1. **release_metrics.csv**: 每个类别的指标
   - `semantic_spread`: 语义分布广度
   - `redundancy_index`: 冗余度指数
   - `cluster_entropy`: 聚类熵

2. **delta_metrics.csv**: 类别间比较指标
   - `change_magnitude`: 变化幅度
   - `novelty_density`: 新颖性密度

3. **检查点文件**: 
   - `*.pkl`: 句子提取检查点
   - `*.npz`: 嵌入向量检查点

## 🧪 测试运行

### 快速测试 (Windows)

```batch
# 使用提供的测试脚本
test_qxdm.bat
```

### 手动测试

```powershell
# 设置测试路径
$env:QXDM_ROOT = "C:\Users\EEzim\Desktop\3gpp_Complexity\test_logs"

# 运行测试
python src\tspec_metrics_2.py --checkpoint-file test_checkpoint.pkl --embeds-file test_embeddings.npz
```

## 📈 监控和日志

### Weights & Biases集成

```bash
# 启用W&B监控 (可选)
python src/tspec_metrics_2.py --wandb-project QXDM-Complexity --log-sys

# 查看实时系统监控
# - CPU/内存使用率
# - GPU利用率和显存
# - 处理进度和性能指标
```

### 进度显示

```
============================================================
📊 System Configuration: NT | 6 threads
📈 Optimized Parameters:
   → Processing Device: cuda
   → Block Size: 500,000 chars
   → Batch Size: 384
   → Processing Mode: Batch
============================================================

🚀 Starting QXDM Log Analysis Pipeline
📄 Stage 1/3: Extracting sentences: 33%|████████▌     | 1/3 stages
📊 Computing metrics for Category1 (13 sentences)
Category metrics: 100%|██████████| 2/2 [00:02<00:00]
✅ Analysis Complete!
```

## 🔍 支持的文件格式

- `.txt` - 文本日志
- `.log` - 日志文件  
- `.csv` - CSV格式数据
- `.json` - JSON格式数据
- `.xml` - XML格式数据
- `.md` - Markdown文档
- `.pdf` - PDF文档 (需要pdfminer)

## ⚡ 性能调优建议

### 大规模数据集

```bash
# 增加块大小以减少内存碎片
export BLOCK_SIZE=1000000

# GPU内存充足时增加批处理大小
export BATCH_SIZE=512

# 启用系统监控
--log-sys --sys-interval 30
```

### 内存受限环境

```bash
# 减少块大小
export BLOCK_SIZE=100000

# 减少批处理大小
export BATCH_SIZE=128

# 使用更小的采样大小进行指标计算
```

## 🐛 故障排除

### Windows 常见问题

1. **spaCy进程挂起**: 代码已自动使用单进程模式
2. **GPU内存不足**: 自动调整批处理大小
3. **路径问题**: 使用绝对路径，确保QXDM_ROOT正确设置

### Linux 常见问题

1. **CUDA版本不匹配**: 确保PyTorch与CUDA版本兼容
2. **内存不足**: 调整BLOCK_SIZE和BATCH_SIZE参数
3. **权限问题**: 确保输出目录可写

## 📞 联系信息

如有问题或建议，请联系：
- 邮箱: ziming.liu@sheffield.ac.uk
- 项目: QXDM Complexity Analysis Pipeline
