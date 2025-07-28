# 简化版QXDM测试数据集

这是一个为测试目的创建的简化版QXDM数据集，包含了原项目的核心结构但数据量更小、更容易管理。

## 📁 数据集结构

```
test_dataset/
├── QXDM_Logs_Sample/          # 简化的QXDM日志样本
│   ├── Urban/                 # 城市场景
│   │   └── urban_test_1.txt
│   ├── Highway/               # 高速公路场景  
│   │   └── highway_test_1.txt
│   └── Rural/                 # 农村场景
│       └── rural_test_1.txt
├── test_logs_sample/          # 测试日志样本
│   ├── RRC_Messages/          # RRC消息日志
│   │   └── rrc_connection_setup.log
│   ├── MAC_Layer/             # MAC层日志
│   │   └── mac_scheduling.log
│   └── PHY_Logs/              # 物理层日志
│       └── phy_measurements.log
├── test_processor.py          # 简化的数据处理脚本
└── README_test_dataset.md     # 本文档
```

## 📊 数据特点

### QXDM日志样本特点:
- **Urban**: 高密度城市环境，多载波聚合，高吞吐量
- **Highway**: 高速移动场景，快速切换，波束管理
- **Rural**: 覆盖增强，低数据速率，功率控制优化

### 测试日志特点:
- **RRC Messages**: 连接建立、测量报告、切换命令
- **MAC Layer**: 调度信息、HARQ反馈、缓冲区状态
- **PHY Logs**: 信道质量、物理层测量、同步信息

## 🚀 使用方法

### 1. 运行简化处理脚本
```bash
cd test_dataset
python test_processor.py
```

### 2. 输出文件
- `test_log_analysis.csv`: 日志文件分析结果
- `sample_metrics.csv`: 生成的性能指标样本

### 3. 主要性能指标
- **RSRP (dBm)**: 参考信号接收功率
- **RSRQ (dB)**: 参考信号接收质量  
- **SINR (dB)**: 信干噪比
- **Throughput (Mbps)**: 上下行吞吐量
- **Latency (ms)**: 时延
- **Packet Loss (%)**: 丢包率

## 📈 测试场景对比

| 场景 | RSRP范围 | 吞吐量 | 特点 |
|------|----------|---------|------|
| Urban | -75 to -95 dBm | 100-200 Mbps | 高密度，干扰复杂 |
| Highway | -70 to -90 dBm | 150-300 Mbps | 高速移动，载波聚合 |
| Rural | -95 to -120 dBm | 5-20 Mbps | 覆盖增强，低速率 |

## 🔧 扩展测试

可以基于此简化数据集进行以下测试:
1. 日志解析算法验证
2. 性能指标计算测试  
3. 机器学习模型训练验证
4. 数据可视化功能测试
5. 批处理流程验证

## 📝 注意事项

- 此数据集仅用于测试和开发目的
- 数据为模拟生成，不包含真实用户信息
- 可根据实际需求调整数据规模和复杂度
- 建议在完整数据集上验证最终结果
