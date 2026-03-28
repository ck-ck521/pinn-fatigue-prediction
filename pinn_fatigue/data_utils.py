# pinn_fatigue/data_utils.py

"""
嵌入式疲劳数据加载和预处理
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from typing import Dict, Optional

# ============================================================
# 嵌入式数据
# ============================================================

# FLT-A 数据
FLT_A_DATA = [
    # [0°层, ±45°层, 90°层, 应力比, 载荷, 寿命]
    [46, 36, 18, 0.1, 847.805, 48],
    [46, 36, 18, 0.1, 847.805, 62],
    [46, 36, 18, 0.1, 844.350, 124],
    [46, 36, 18, 0.1, 846.078, 192],
    [46, 36, 18, 0.1, 846.078, 258],
    [46, 36, 18, 0.1, 742.428, 441],
    [46, 36, 18, 0.1, 742.428, 636],
    [46, 36, 18, 0.1, 743.810, 1125],
    [46, 36, 18, 0.1, 743.810, 1710],
    [46, 36, 18, 0.1, 743.810, 2545],
    [46, 36, 18, 0.1, 693.712, 756],
    [46, 36, 18, 0.1, 693.712, 2412],
    [46, 36, 18, 0.1, 693.712, 4435],
    [46, 36, 18, 0.1, 693.712, 6600],
    [46, 36, 18, 0.1, 693.712, 13708],
    [46, 36, 18, 0.1, 645.342, 8272],
    [46, 36, 18, 0.1, 643.614, 11920],
    [46, 36, 18, 0.1, 643.614, 20402],
    [46, 36, 18, 0.1, 643.614, 59762],
    [46, 36, 18, 0.1, 643.614, 92190],
    [46, 36, 18, 0.1, 593.171, 163539],
    [46, 36, 18, 0.1, 594.899, 239937],
    [46, 36, 18, 0.1, 594.899, 208650],
    [46, 36, 18, 0.1, 594.899, 715537],
    [46, 36, 18, 0.1, 593.171, 1103799],
    [46, 36, 18, 0.1, 544.801, 959869],
    [46, 36, 18, 0.1, 546.529, 1642820],
    [46, 36, 18, 0.1, 546.529, 2722483],
    [46, 36, 18, 0.1, 548.256, 4051958],
]

# FLT-B 数据
FLT_B_DATA = [
    [40, 40, 20, 0.1, 741.675, 80],
    [40, 40, 20, 0.1, 744.218, 123],
    [40, 40, 20, 0.1, 746.761, 157],
    [40, 40, 20, 0.1, 744.218, 290],
    [40, 40, 20, 0.1, 700.627, 433],
    [40, 40, 20, 0.1, 700.627, 580],
    [40, 40, 20, 0.1, 703.170, 891],
    [40, 40, 20, 0.1, 657.399, 1226],
    [40, 40, 20, 0.1, 657.399, 2267],
    [40, 40, 20, 0.1, 654.856, 3482],
    [40, 40, 20, 0.1, 616.351, 2969],
    [40, 40, 20, 0.1, 613.808, 6793],
    [40, 40, 20, 0.1, 616.351, 9134],
    [40, 40, 20, 0.1, 616.351, 13252],
    [40, 40, 20, 0.1, 526.989, 125459],
    [40, 40, 20, 0.1, 521.904, 272212],
    [40, 40, 20, 0.1, 526.989, 982391],
    [40, 40, 20, 0.1, 526.989, 1252631],
    [40, 40, 20, 0.1, 480.856, 622872],
    [40, 40, 20, 0.1, 480.856, 907098],
    [40, 40, 20, 0.1, 478.313, 2379644],
    [40, 40, 20, 0.1, 480.856, 3361825],
    [40, 40, 20, 0.1, 480.856, 5591744],
    [40, 40, 20, 0.1, 437.265, 1063933],
]

# FLT-C 数据
FLT_C_DATA = [
    [30, 60, 10, 0.1, 581.125, 2718],
    [30, 60, 10, 0.1, 544.349, 2718],
    [30, 60, 10, 0.1, 546.895, 3597],
    [30, 60, 10, 0.1, 582.257, 1495],
    [30, 60, 10, 0.1, 545.764, 7636],
    [30, 60, 10, 0.1, 512.948, 3297],
    [30, 60, 10, 0.1, 510.119, 8489],
    [30, 60, 10, 0.1, 510.119, 12253],
    [30, 60, 10, 0.1, 476.172, 3721],
    [30, 60, 10, 0.1, 477.587, 12487],
    [30, 60, 10, 0.1, 474.758, 20037],
    [30, 60, 10, 0.1, 441.942, 15144],
    [30, 60, 10, 0.1, 444.771, 104276],
    [30, 60, 10, 0.1, 443.357, 173121],
    [30, 60, 10, 0.1, 444.771, 304201],
    [30, 60, 10, 0.1, 443.357, 624216],
    [30, 60, 10, 0.1, 409.127, 135379],
    [30, 60, 10, 0.1, 409.127, 402480],
    [30, 60, 10, 0.1, 407.995, 1397329],
    [30, 60, 10, 0.1, 376.594, 402480],
    [30, 60, 10, 0.1, 376.594, 532510],
    [30, 60, 10, 0.1, 376.594, 1986552],
]


def get_all_data() -> list:
    """获取所有数据（带组别标签）"""
    data = []
    for d in FLT_A_DATA:
        data.append(['FLT-A'] + d)
    for d in FLT_B_DATA:
        data.append(['FLT-B'] + d)
    for d in FLT_C_DATA:
        data.append(['FLT-C'] + d)
    return data


def get_dataframe() -> pd.DataFrame:
    """获取完整的DataFrame"""
    data = get_all_data()
    columns = ['组别', '0°层含量(%)', '±45°层含量(%)', '90°层含量(%)',
               '应力比(R)', '拉伸疲劳载荷(MPa)', '疲劳寿命']
    return pd.DataFrame(data, columns=columns)


# ============================================================
# 数据加载函数
# ============================================================

def load_data(file_path=None, test_size: float = 0.5, random_state: int = 42) -> Dict:
    """
    从嵌入式数据加载并预处理疲劳数据。

    Args:
        file_path: 可选，如果提供则从文件加载（保留兼容性）
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        包含训练/测试数据的字典
    """
    # 使用嵌入式数据创建DataFrame
    df = get_dataframe()

    # 确保载荷为正
    df['拉伸疲劳载荷(MPa)'] = df['拉伸疲劳载荷(MPa)'].clip(lower=1e-6)

    # 准备特征和目标变量
    X = df[['0°层含量(%)', '±45°层含量(%)', '90°层含量(%)',
            '应力比(R)', '拉伸疲劳载荷(MPa)']].values
    y = np.log10(df['疲劳寿命'].values.reshape(-1, 1) + 1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).requires_grad_(True)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    print(f"[数据加载] 从嵌入式数据加载 {len(df)} 个样本")
    print(f"  FLT-A: {len(df[df['组别'] == 'FLT-A'])} 样本")
    print(f"  FLT-B: {len(df[df['组别'] == 'FLT-B'])} 样本")
    print(f"  FLT-C: {len(df[df['组别'] == 'FLT-C'])} 样本")
    print(f"  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    return {
        'df': df,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_scaled,
        'y_test': y_test_scaled,
        'X_train_tensor': X_train_tensor,
        'X_test_tensor': X_test_tensor,
        'y_train_tensor': y_train_tensor,
        'y_test_tensor': y_test_tensor,
        'train_loader': train_loader,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }


# 兼容接口
prepare_data = load_data