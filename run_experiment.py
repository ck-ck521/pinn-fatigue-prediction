"""
PINN 疲劳寿命预测完整实验脚本

直接运行即可，无需外部Excel文件（数据已嵌入代码中）
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 导入模块
from pinn_fatigue import (
    PhysicsInformedNeuralNetwork,
    load_data,
    train_model,
    evaluate_model,
    Config
)


def print_section(title):
    """打印分隔标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """打印子标题"""
    print(f"\n--- {title} ---")


def main():
    """主函数：执行完整实验"""

    # ============================================================
    # 1. 实验配置
    # ============================================================
    print_section("PINN 疲劳寿命预测模型实验")
    print(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建配置
    config = Config()
    config.save_dir = "results"
    config.test_size = 0.5
    config.epochs = 1000
    config.learning_rate = 1e-3
    config.lambda_phys = 2.0
    config.batch_size = 16

    print(f"\n实验配置:")
    print(f"  测试集比例: {config.test_size}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  物理损失权重: {config.lambda_phys}")
    print(f"  批量大小: {config.batch_size}")
    print(f"  结果保存目录: {config.save_dir}")

    # ============================================================
    # 2. 加载数据（嵌入式数据，无需外部文件）
    # ============================================================
    print_section("数据加载")

    data = load_data()  # 自动使用嵌入式数据

    # 打印数据统计
    df = data['df']
    print(f"\n数据统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  训练集: {len(data['X_train'])} 样本")
    print(f"  测试集: {len(data['X_test'])} 样本")

    # 按组别统计
    if '组别' in df.columns:
        print(f"\n按组别分布:")
        for group in ['FLT-A', 'FLT-B', 'FLT-C']:
            count = len(df[df['组别'] == group])
            print(f"    {group}: {count} 样本")

    # 数据范围
    print(f"\n数据范围:")
    print(f"  载荷: {df['拉伸疲劳载荷(MPa)'].min():.2f} ~ {df['拉伸疲劳载荷(MPa)'].max():.2f} MPa")
    print(f"  寿命: {df['疲劳寿命'].min():.0f} ~ {df['疲劳寿命'].max():.0f} 循环")

    # ============================================================
    # 3. 创建模型
    # ============================================================
    print_section("模型创建")

    model = PhysicsInformedNeuralNetwork(
        input_dim=5,           # 5个特征：w0, w45, w90, R, S
        hidden_dim=32,
        use_dropout=False
    )

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型信息:")
    print(f"  输入维度: 5 (0°层, ±45°层, 90°层, 应力比, 载荷)")
    print(f"  隐藏层维度: 32")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 打印初始物理参数
    print(f"\n初始物理参数:")
    print(f"  α (截距): {model.alpha.item():.6f}")
    print(f"  β₁ (0°层): {model.beta1.item():.6f}")
    print(f"  β₂ (±45°层): {model.beta2.item():.6f}")
    print(f"  β₃ (90°层): {model.beta3.item():.6f}")
    print(f"  γ (载荷系数): {model.gamma.item():.6f}")

    # ============================================================
    # 4. 训练模型
    # ============================================================
    print_section("模型训练")

    model, history = train_model(
        model, data, config,
        model_name='pinn_model'
    )

    # 打印训练结果
    print(f"\n训练完成!")
    print(f"  最终训练损失: {history['train_losses'][-1]:.6f}")
    print(f"  最佳验证损失: {min(history['test_losses']):.6f}")

    # 打印最终物理参数
    final_params = model.get_physical_params()
    print(f"\n最终物理参数:")
    print(f"  α (截距): {final_params['alpha']:.6f}")
    print(f"  β₁ (0°层): {final_params['beta1']:.6f} {'↑' if final_params['beta1'] > 0 else '↓'}")
    print(f"  β₂ (±45°层): {final_params['beta2']:.6f} {'↑' if final_params['beta2'] > 0 else '↓'}")
    print(f"  β₃ (90°层): {final_params['beta3']:.6f} {'↓' if final_params['beta3'] < 0 else '↑'}")
    print(f"  γ (载荷系数): {final_params['gamma']:.6f}")

    # 物理参数合理性检查
    print(f"\n物理参数合理性:")
    print(f"  β₁ > 0 (0°层正向影响): {'✓' if final_params['beta1'] > 0 else '✗'}")
    print(f"  β₂ > 0 (±45°层正向影响): {'✓' if final_params['beta2'] > 0 else '✗'}")
    print(f"  β₃ < 0 (90°层负向影响): {'✓' if final_params['beta3'] < 0 else '✗'}")
    print(f"  γ > 0 (载荷负向影响): {'✓' if final_params['gamma'] > 0 else '✗'}")

    # ============================================================
    # 5. 评估模型
    # ============================================================
    print_section("模型评估")

    metrics = evaluate_model(
        model, data, config.save_dir,
        model_name='pinn_model'
    )

    # 打印评估结果
    print(f"\n评估指标:")
    print(f"\n原始尺度:")
    print(f"  R² = {metrics['r2_original']:.4f}")
    print(f"  RMSE = {metrics['rmse_original']:.2f}")
    print(f"  MAE = {metrics['mae_original']:.2f}")
    print(f"  平均相对误差 = {metrics['mre_original']:.2f}%")

    print(f"\n对数尺度:")
    print(f"  R² = {metrics['r2_log']:.4f}")
    print(f"  RMSE = {metrics['rmse_log']:.4f}")
    print(f"  MAE = {metrics['mae_log']:.4f}")
    print(f"  平均相对误差 = {metrics['mre_log']:.2f}%")

    # ============================================================
    # 6. 生成预测对比表
    # ============================================================
    print_section("预测对比")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    # 对测试集进行预测
    with torch.no_grad():
        y_pred_scaled = model(data['X_test_tensor'].to(device)).cpu().numpy()

    y_pred_phys = 10 ** data['scaler_y'].inverse_transform(y_pred_scaled) - 1
    y_test_phys = 10 ** data['scaler_y'].inverse_transform(data['y_test']) - 1

    # 创建预测对比表
    comparison_df = pd.DataFrame({
        '真实寿命': y_test_phys.flatten(),
        '预测寿命': y_pred_phys.flatten(),
        '绝对误差': np.abs(y_pred_phys.flatten() - y_test_phys.flatten()),
        '相对误差(%)': np.abs((y_pred_phys.flatten() - y_test_phys.flatten()) / y_test_phys.flatten() * 100)
    })

    comparison_df = comparison_df.sort_values('真实寿命')

    print(f"\n测试集预测对比（前10条）:")
    print(comparison_df.head(10).to_string(index=False))

    # 保存完整预测对比表
    comparison_path = os.path.join(config.save_dir, 'metrics', 'prediction_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n完整预测对比表已保存: {comparison_path}")

    # 统计误差分布
    errors = comparison_df['相对误差(%)'].values
    print(f"\n误差分布统计:")
    print(f"  平均误差: {np.mean(errors):.2f}%")
    print(f"  中位数误差: {np.median(errors):.2f}%")
    print(f"  标准差: {np.std(errors):.2f}%")
    print(f"  最大误差: {np.max(errors):.2f}%")
    print(f"  最小误差: {np.min(errors):.2f}%")

    # 计算误差在20%以内的比例
    within_20 = np.sum(errors <= 20) / len(errors) * 100
    within_50 = np.sum(errors <= 50) / len(errors) * 100
    print(f"\n  误差 ≤ 20%: {within_20:.1f}%")
    print(f"  误差 ≤ 50%: {within_50:.1f}%")

    # ============================================================
    # 7. 保存训练历史
    # ============================================================
    print_section("保存结果")

    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': history['epochs'],
        'train_loss': history['train_losses'],
        'test_loss': history['test_losses'],
        'data_loss': history['data_losses'],
        'physics_loss': history['physics_losses']
    })
    history_path = os.path.join(config.save_dir, 'loss_data', 'training_history.csv')
    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    print(f"训练历史已保存: {history_path}")

    # 保存物理参数演化
    params_history = history['physical_params_history']
    params_df = pd.DataFrame({
        'epoch': history['epochs'],
        'alpha': params_history['alpha'],
        'beta1': params_history['beta1'],
        'beta2': params_history['beta2'],
        'beta3': params_history['beta3'],
        'gamma': params_history['gamma']
    })
    params_path = os.path.join(config.save_dir, 'plot_data', 'physical_params_history.csv')
    params_df.to_csv(params_path, index=False, encoding='utf-8-sig')
    print(f"物理参数历史已保存: {params_path}")

    # 保存最终模型
    model_path = os.path.join(config.save_dir, 'models', 'pinn_model_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存: {model_path}")

    # 保存实验摘要
    summary = {
        '实验时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '总样本数': len(data['df']),
        '训练样本数': len(data['X_train']),
        '测试样本数': len(data['X_test']),
        '训练轮数': config.epochs,
        '学习率': config.learning_rate,
        '物理损失权重': config.lambda_phys,
        '最终训练损失': history['train_losses'][-1],
        '最佳验证损失': min(history['test_losses']),
        'R²原始尺度': metrics['r2_original'],
        'RMSE原始尺度': metrics['rmse_original'],
        '平均相对误差原始尺度': metrics['mre_original'],
        'R²对数尺度': metrics['r2_log'],
        'RMSE对数尺度': metrics['rmse_log'],
        '平均相对误差对数尺度': metrics['mre_log'],
        'α': final_params['alpha'],
        'β₁': final_params['beta1'],
        'β₂': final_params['beta2'],
        'β₃': final_params['beta3'],
        'γ': final_params['gamma']
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(config.save_dir, 'metrics', 'experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"实验摘要已保存: {summary_path}")

    # ============================================================
    # 8. 绘制误差直方图
    # ============================================================
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(errors), color='red', linestyle='--',
                linewidth=2, label=f'均值: {np.mean(errors):.2f}%')
    plt.axvline(np.median(errors), color='green', linestyle='--',
                linewidth=2, label=f'中位数: {np.median(errors):.2f}%')
    plt.xlabel('相对误差 (%)')
    plt.ylabel('频数')
    plt.title('测试集预测误差分布')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(config.save_dir, 'figures', 'error_distribution.png'), dpi=300)
    plt.close()
    print(f"误差分布图已保存: {config.save_dir}/figures/error_distribution.png")

    # ============================================================
    # 9. 最终总结
    # ============================================================
    print_section("实验完成")

    print(f"\n实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n所有结果已保存到: {os.path.abspath(config.save_dir)}/")
    print(f"\n文件结构:")
    print(f"  ├── models/           # 模型权重文件")
    print(f"  ├── figures/          # 可视化图表")
    print(f"  ├── metrics/          # 评估指标CSV")
    print(f"  ├── loss_data/        # 训练损失数据")
    print(f"  └── plot_data/        # 绘图数据")

    print("\n" + "=" * 70)
    print("实验成功完成！")
    print("=" * 70)

    return model, data, history, metrics


if __name__ == "__main__":
    try:
        result = main()
        print("\n✓ 实验运行成功！")
    except Exception as e:
        print(f"\n✗ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()