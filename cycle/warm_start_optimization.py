
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from physics_engine import optimize_GRE_DE, global_objective, simulate_cost, plot_results, build_rider_profile, PhysParams, OptParams
import cma
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def optimize_with_warm_start(csv_path, rider_type, sex, phys, opt, model_path='ag_power_predictor'):
    """
    使用AutoGluon模型的预测结果作为CMA-ES优化的初始种群中心 (Warm Start)
    """
    df = pd.read_csv(csv_path)
    ds_vec = df["Distance_step_m"].values
    grade_col = "Grade_pct_smooth" if "Grade_pct_smooth" in df.columns else "Grade_pct"
    grade_vec = df[grade_col].values / 100.0
    wind_vec = np.zeros(len(ds_vec))

    rider = build_rider_profile(rider_type, sex)
    N = len(ds_vec)
    K = int(np.ceil(N / opt.seg_stride))
    
    print(f"正在加载AutoGluon模型: {model_path} ...")
    try:
        predictor = TabularPredictor.load(model_path)
        
        # 构造特征进行预测
        pred_data = pd.DataFrame({
            "Grade": df[grade_col],
            "Distance": df["Distance_m"],
            "Altitude": df["Altitude_m"]
        })
        
        print("生成初始猜测 (Warm Start Prediction)...")
        P_pred = predictor.predict(pred_data)
        
        # 将功率转换为rho (0.1 - 1.0)
        # Power = Pmin + rho * (Pmax - Pmin)
        # rho = (Power - Pmin) / (Pmax - Pmin)
        Pmax = min(opt.Pmax_abs, rider.Pmax_w)
        rho_pred = (P_pred.values - opt.Pmin) / (Pmax - opt.Pmin)
        rho_pred = np.clip(rho_pred, 0.1, 1.0)
        
        # 下采样到K个变量 (取每个段的平均值)
        x0_warm = []
        for i in range(K):
            start_idx = i * opt.seg_stride
            end_idx = min((i + 1) * opt.seg_stride, N)
            segment_rhos = rho_pred[start_idx:end_idx]
            x0_warm.append(np.mean(segment_rhos))
        
        x0 = np.array(x0_warm)
        print("初始猜测生成完毕。")
        sigma0 = 0.05 # 既然我们信任预测，可以减小搜索方差 (原为0.2)
        
    except Exception as e:
        print(f"加载模型或生成预测失败: {e}。回退到默认初始化。")
        x0 = 0.5 * np.ones(K)
        sigma0 = 0.2

    # --- 开始优化 (复制自 physics_engine.py 但使用了新的 x0) ---
    
    cma_opts = {
        'bounds': [0.1, 1.0],
        'popsize': opt.popsize,
        'maxiter': opt.maxiter,
        'verbose': 0,
        'seed': opt.seed if opt.seed is not None else np.random.randint(2**30),
    }

    print(f"开始优化(CMA-ES Warm Start): 变量数={K}, 初始cost={global_objective(x0, ds_vec, grade_vec, wind_vec, rider, phys, opt, N, opt.seg_stride):.2f}")

    res = cma.fmin(global_objective, x0, sigma0, options=cma_opts,
                   args=(ds_vec, grade_vec, wind_vec, rider, phys, opt, N, opt.seg_stride))
    
    x_final = res[0]

    # polish with L-BFGS-B
    lb = 0.1 * np.ones(K)
    ub = 1.0 * np.ones(K)
    res_lbfgs = minimize(global_objective, x_final, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                         args=(ds_vec, grade_vec, wind_vec, rider, phys, opt, N, opt.seg_stride),
                         options={'maxiter': 200, 'ftol': 1e-6})
    x_final = res_lbfgs.x
    print(f"最终f-value: {res_lbfgs.fun:.4f}")
    
    rho_opt = np.repeat(x_final, opt.seg_stride)[:N]
    cost, diag = simulate_cost(rho_opt, ds_vec, grade_vec, wind_vec, rider, phys, opt)
    P_opt = opt.Pmin + rho_opt * (min(opt.Pmax_abs, rider.Pmax_w) - opt.Pmin)

    # 绘图保存为不同名字
    plot_results_warm(ds_vec, grade_vec, P_opt, rider, "optimized_power_warm_start.png")
    
    return res_lbfgs.fun

def plot_results_warm(ds_vec, grade_vec, P_opt, rider_info, filename):
    distance_km = np.cumsum(ds_vec) / 1000.0
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:green' # Different color
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Power Output (W)', color=color1)
    ax1.plot(distance_km, P_opt, color=color1, label='Optimized Power (Warm Start)', linewidth=1.5)
    ax1.axhline(y=rider_info.CP_w, color='r', linestyle='--', label='CP')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()
    color2 = 'tab:gray'
    ax2.set_ylabel('Grade (%)', color=color2)
    ax2.fill_between(distance_km, 0, grade_vec * 100, color=color2, alpha=0.2)
    plt.title(f'Warm Start Optimization Result')
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

if __name__ == "__main__":
    CSV_FILE = "./route_with_grade.csv"
    phys_cfg = PhysParams()
    # 使用相同的配置以便对比
    opt_cfg = OptParams(
        seg_stride=20,
        maxiter=200,
        popsize=8,
        seed=42 # 固定种子对比 CMA 内部随机性
    )

    print("--- Baseline Run (Standard Init) ---")
    # 为了公平对比，我们需要在同一个脚本里跑一次默认的
    # 调用原版函数(如果在 physics_engine 里没改的话，它用 0.5)
    from physics_engine import optimize_GRE_DE
    res_base, _, _, _ = optimize_GRE_DE(CSV_FILE, "Time trialist", "Men", phys_cfg, opt_cfg)
    base_score = res_base.fun
    
    print("\n--- Warm Start Run (AutoGluon Init) ---")
    warm_score = optimize_with_warm_start(CSV_FILE, "Time trialist", "Men", phys_cfg, opt_cfg)

    print("\n" + "="*40)
    print(f"Baseline Score:   {base_score:.4f}")
    print(f"Warm Start Score: {warm_score:.4f}")
    print(f"Improvement:      {base_score - warm_score:.4f}")
    print("="*40)
