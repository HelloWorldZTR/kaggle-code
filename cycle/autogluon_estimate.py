
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor, TabularDataset
from physics_engine import optimize_GRE_DE, PhysParams, OptParams

def run_autogluon_estimation():
    # 1. 运行物理模型优化以获取"Ground Truth"数据
    print("正在运行物理模型优化以生成训练数据...")
    CSV_FILE = "./route_with_grade.csv"
    phys_cfg = PhysParams()
    # 为了演示速度，减少迭代次数
    opt_cfg = OptParams(
        seg_stride=20,
        maxiter=200,   # 稍微减少迭代次数以加快速度
        popsize=8,
        seed=42
    )

    try:
        # 这里会调用 cma.plot()，可能会生成一些图片文件，我们可以忽略
        res, diags, P_opt_truth, r_info = optimize_GRE_DE(
            CSV_FILE, "Time trialist", "Men", phys_cfg, opt_cfg
        )
    except FileNotFoundError:
        print(f"找不到 {CSV_FILE}")
        return

    print("物理优化完成。准备AutoGluon数据集...")

    # 2. 构建数据集
    df_route = pd.read_csv(CSV_FILE)
    
    # 确保长度一致
    N = len(P_opt_truth)
    df_route = df_route.iloc[:N].copy()

    # 构造特征
    # 我们希望模型根据路况预测功率
    # 特征工程: 坡度, 距离, 剩余距离
    grade_col = "Grade_pct_smooth" if "Grade_pct_smooth" in df_route.columns else "Grade_pct"
    
    data = pd.DataFrame({
        "Grade": df_route[grade_col],
        "Distance": df_route["Distance_m"],
        "Altitude": df_route["Altitude_m"],
        # 添加一些简单的上下文特征，如前方路况
        # "Grade_Next_100m": df_route[grade_col].rolling(window=10, center=False).mean().shift(-10).fillna(0) # 简单平均
    })
    
    # 添加目标变量
    data["Power_Target"] = P_opt_truth

    # 划分训练集和测试集
    # 对于时间序列/路线数据，通常不应该随机划分，而是应该分段。
    # 但为了简单演示 "学习这种策略"，我们随机划分，或者拿另一段路测试。
    # 这里我们采用随机划分(80/20)，假设是在学习 "在某种坡度下的出力习惯"
    train_data = TabularDataset(data.sample(frac=0.8, random_state=42))
    test_data = TabularDataset(data.drop(train_data.index))

    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

    # 3. 使用AutoGluon训练
    save_path = 'ag_power_predictor'
    predictor = TabularPredictor(label='Power_Target', path=save_path, problem_type='regression').fit(
        train_data,
        time_limit=120, # 2分钟训练时间
        presets='medium_quality'
    )

    # 4. 评估
    print("\n评估结果:")
    performance = predictor.evaluate(test_data)
    print(performance)

    # 5. 在全路段上预测并绘图对比
    print("\n生成全路段预测...")
    data_full = TabularDataset(data.drop(columns=["Power_Target"]))
    P_pred = predictor.predict(data_full)

    # 绘图对比
    plt.figure(figsize=(12, 6))
    
    dist_km = data["Distance"] / 1000.0
    
    plt.plot(dist_km, data["Power_Target"], label='Physics Optimized (Truth)', alpha=0.7, color='blue')
    plt.plot(dist_km, P_pred, label='AutoGluon Predicted', alpha=0.7, color='orange', linestyle='--')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Power (W)')
    plt.title('Power Profile: Physics Model vs AutoGluon Estimator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_img = "autogluon_comparison.png"
    plt.savefig(out_img, dpi=150)
    print(f"对比图已保存至: {out_img}")

    # 特征重要性
    print("\n特征重要性:")
    try:
        importance = predictor.feature_importance(test_data)
        print(importance)
    except Exception as e:
        print(f"无法计算特征重要性: {e}")

if __name__ == "__main__":
    run_autogluon_estimation()
