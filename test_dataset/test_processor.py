#!/usr/bin/env python3
"""
简化版QXDM测试数据处理脚本
Simplified QXDM Test Data Processor
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def process_test_logs(log_dir):
    """处理测试日志文件"""
    results = []
    
    for log_file in Path(log_dir).rglob("*.txt"):
        if log_file.is_file():
            print(f"Processing: {log_file}")
            
            # 简单的日志解析
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 提取基本信息
            file_info = {
                'file': log_file.name,
                'location': log_file.parent.name,
                'size_kb': log_file.stat().st_size / 1024,
                'line_count': len(content.splitlines()),
                'contains_rrc': 'RRC' in content,
                'contains_mac': 'MAC' in content,
                'contains_phy': 'PHY' in content,
                'rsrp_mentions': content.count('RSRP'),
                'throughput_mentions': content.count('Throughput')
            }
            results.append(file_info)
    
    return pd.DataFrame(results)

def generate_sample_metrics():
    """生成示例性能指标"""
    scenarios = ['Urban', 'Highway', 'Rural']
    metrics = []
    
    for scenario in scenarios:
        for i in range(5):  # 每个场景5个样本
            metric = {
                'scenario': scenario,
                'test_id': f"{scenario}_{i+1}",
                'avg_rsrp_dbm': np.random.normal(-85, 10),
                'avg_rsrq_db': np.random.normal(-10, 3),
                'avg_sinr_db': np.random.normal(15, 5),
                'dl_throughput_mbps': np.random.normal(100, 30),
                'ul_throughput_mbps': np.random.normal(50, 15),
                'latency_ms': np.random.normal(20, 8),
                'packet_loss_percent': np.random.exponential(0.5),
                'handover_count': np.random.poisson(2),
                'connection_time_s': np.random.exponential(300)
            }
            metrics.append(metric)
    
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    print("=== 简化版QXDM测试数据处理器 ===")
    
    # 处理测试日志
    test_log_dir = "./QXDM_Logs_Sample"
    if os.path.exists(test_log_dir):
        print(f"\n处理测试日志目录: {test_log_dir}")
        log_results = process_test_logs(test_log_dir)
        print("\n日志处理结果:")
        print(log_results.to_string(index=False))
        
        # 保存结果
        log_results.to_csv("test_log_analysis.csv", index=False)
        print("\n日志分析结果已保存到: test_log_analysis.csv")
    
    # 生成示例指标
    print("\n生成示例性能指标...")
    metrics_df = generate_sample_metrics()
    print("\n性能指标预览:")
    print(metrics_df.head(10).to_string(index=False))
    
    # 保存指标
    metrics_df.to_csv("sample_metrics.csv", index=False)
    print("\n示例指标已保存到: sample_metrics.csv")
    
    # 基本统计
    print("\n=== 基本统计信息 ===")
    print(f"总测试文件数: {len(log_results) if 'log_results' in locals() else 0}")
    print(f"生成指标记录数: {len(metrics_df)}")
    
    print("\n各场景平均性能:")
    scenario_stats = metrics_df.groupby('scenario').agg({
        'avg_rsrp_dbm': 'mean',
        'dl_throughput_mbps': 'mean',
        'latency_ms': 'mean'
    }).round(2)
    print(scenario_stats.to_string())
    
    print("\n测试数据集创建完成！")
