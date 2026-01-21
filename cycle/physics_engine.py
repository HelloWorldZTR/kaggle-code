from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from scipy.optimize import minimize
import cma
from numba import njit

# =========================
# 1) 核心生理模型函数 (Numba Optimized)
# =========================
A_FATIGUE = 50.0
T_CP_MAX = 1800.0

@njit
def ompd_power(t, Pmax, CP, Wp, A=A_FATIGUE, Tcpmax=T_CP_MAX):
    # t can be scalar or array
    t_val = np.maximum(t, 1e-6)
    core = (Wp / t_val) * (1 - np.exp(-t_val * (Pmax - CP) / Wp)) + CP
    # numpy where handles arrays, but for scalar t in loops we might want scalar logic
    # However ompd_power is called with 'dt' which is scalar in the loop below.
    # But later it might be used differently. The original code used np.where.
    return np.where(t_val > Tcpmax, core - A * np.log(t_val / Tcpmax), core)

@njit
def softplus(x, k=10.0):
    kx = k * x
    # Scalar optimization, assuming x is scalar inside loop
    if kx > 50.0:
        return x
    
    # Manual clip for scalar (Numba np.clip sometimes problematic with scalars)
    val = kx
    if val > 500.0:
        val = 500.0
    elif val < -500.0:
        val = -500.0
        
    return np.log1p(np.exp(val)) / k


# =========================
# 2) 数据结构与预设
# =========================
@dataclass
class RiderProfile:
    rider_type: str
    sex: str
    mass_kg: float
    cda: float
    CP_w: float
    Wp_j: float
    Pmax_w: float


@dataclass
class PhysParams:
    crr: float = 0.003
    rho_air: float = 1.17
    g: float = 9.80
    eta: float = 0.975
    bike_kg: float = 8.5


@dataclass
class OptParams:
    Pmin: float = 0.0
    Pmax_abs: float = 900.0
    Pc_NP: float = None
    mu_ompd: float = 1.8e-4
    penalty_big: float = 1e4
    smooth_k: float = 15.0
    seg_stride: int = 50
    maxiter: int = 100
    popsize: int = 6
    seed: Optional[int] = None  # <-- 改为可选；None 表示随机（每次可能不同）
    # 如果想要强制可复现，把 seed 设为固定整数


BASE_GEOM = {
    "Time trialist": {"Men": {"mass_kg": 65, "cda": 0.21}, "Women": {"mass_kg": 60.0, "cda": 0.19}},
    "Sprinter": {"Men": {"mass_kg": 80.2, "cda": 0.23}, "Women": {"mass_kg": 68.0, "cda": 0.21}}
}
CP_WP_BY_SEX = {
    "Time trialist": {"Men": {"cp_w": 400, "w_prime_kj": 14.129}, "Women": {"cp_w": 279.0, "w_prime_kj": 11.892}},
    "Sprinter": {"Men": {"cp_w": 250.4, "w_prime_kj": 23.0}, "Women": {"cp_w": 210.0, "w_prime_kj": 17.18}}
}
P5S_WKG = {"Men": 24.04, "Women": 19.42}


def build_rider_profile(rider_type: str, sex: str) -> RiderProfile:
    geom = BASE_GEOM[rider_type][sex]
    perf = CP_WP_BY_SEX[rider_type][sex]
    mass = geom["mass_kg"]
    Pmax = P5S_WKG[sex] * mass
    return RiderProfile(rider_type, sex, mass, geom["cda"], perf["cp_w"], perf["w_prime_kj"] * 1000.0, Pmax)


# =========================
# 3) 物理仿真核心
# =========================
@njit
def _solve_speed_brentq(P_wheel, grade, wind, mass_total, cda, crr, rho_air, g):
    theta = np.arctan(grade)
    
    # Precompute constants
    k_grav = mass_total * g * np.sin(theta)
    k_roll = crr * mass_total * g * np.cos(theta)
    k_aero = 0.5 * rho_air * cda
    
    # Target function f(v) = Power_required - P_input
    # We inline the logic for speed.
    
    low = 0.1
    high = 50.0
    
    # Check bounds first (optimization)
    # f(0.1)
    v_low = low
    f_low = (k_grav * v_low + k_roll * v_low + k_aero * (v_low - wind)**2 * v_low) - P_wheel
    if f_low > 0.0:
        return low
        
    # f(50.0)
    v_high = high
    f_high = (k_grav * v_high + k_roll * v_high + k_aero * (v_high - wind)**2 * v_high) - P_wheel
    if f_high < 0.0:
        return high

    # Bisection
    for _ in range(30): # 30 iterations is plenty for float precision
        mid = (low + high) * 0.5
        if high - low < 1e-5:
            return mid
            
        f_mid = (k_grav * mid + k_roll * mid + k_aero * (mid - wind)**2 * mid) - P_wheel
        
        if f_mid == 0.0:
            return mid
        
        if f_mid * f_low > 0.0:
            low = mid
            f_low = f_mid
        else:
            high = mid
            # f_high = f_mid
            
    return (low + high) * 0.5

@njit
def _simulate_cost_jit(rho_full, ds_vec, grade_vec, wind_vec, 
                       Pmax_w, CP_w, Wp_j, cda, mass_kg,
                       crr, rho_air, g, eta, bike_kg,
                       Pmin, Pmax_abs, mu_ompd, penalty_big, smooth_k, Pc_target_val, has_pc_target):
    
    N = len(ds_vec)
    Pmax_eff = min(Pmax_abs, Pmax_w)
    P_h = Pmin + rho_full * (Pmax_eff - Pmin)
    P_wheel_vec = eta * P_h
    mass_total = mass_kg + bike_kg

    T = 0.0
    W = Wp_j
    np4_acc = 0.0
    pen_ompd = 0.0
    pen_cons = 0.0

    for i in range(N):
        v = _solve_speed_brentq(P_wheel_vec[i], grade_vec[i], wind_vec[i], 
                                mass_total, cda, crr, rho_air, g)
        
        # Avoid division by zero
        if v < 1e-4:
            v = 1e-4
            
        dt = ds_vec[i] / v
        T += dt
        
        # W' balance
        W += (CP_w - P_h[i]) * dt
        if W < 0:
            pen_cons += ((-W) / 1000.0) ** 2 * penalty_big
        if W > Wp_j:
            W = Wp_j
            
        # OMPD penalty
        Pcap = ompd_power(dt, Pmax_w, CP_w, Wp_j)
        dP = softplus(P_h[i] - Pcap, smooth_k)
        pen_ompd += (dP ** 2) * dt
        
        np4_acc += (P_h[i] ** 4) * dt

    NP = (np4_acc / T) ** 0.25
    
    Pc = Pc_target_val if has_pc_target else CP_w
    if NP > Pc:
        pen_cons += ((NP - Pc) / 10.0) ** 2 * penalty_big

    cost = T + mu_ompd * pen_ompd + pen_cons
    return cost, T, NP, W

def solve_speed_from_power(P_wheel, grade, wind, mass_total, cda, phys: PhysParams):
    # Wrapper for compatibility if needed elsewhere, though main loop uses JIT now.
    return _solve_speed_brentq(P_wheel, grade, wind, mass_total, cda, phys.crr, phys.rho_air, phys.g)

def simulate_cost(rho_full, ds_vec, grade_vec, wind_vec, rider, phys, opt):
    # Unpack for JIT
    Pc_target_val = opt.Pc_NP if opt.Pc_NP is not None else -1.0
    has_pc_target = (opt.Pc_NP is not None)
    
    cost, T, NP, W_end = _simulate_cost_jit(
        rho_full, ds_vec, grade_vec, wind_vec,
        rider.Pmax_w, rider.CP_w, rider.Wp_j, rider.cda, rider.mass_kg,
        phys.crr, phys.rho_air, phys.g, phys.eta, phys.bike_kg,
        opt.Pmin, opt.Pmax_abs, opt.mu_ompd, opt.penalty_big, opt.smooth_k, 
        float(Pc_target_val), has_pc_target
    )
    
    return cost, {"T_sec": T, "NP_W": NP, "W_end_J": W_end}


# =========================
# 4) 优化与绘图
# =========================
def global_objective(rhoK, ds_vec, grade_vec, wind_vec, rider, phys, opt, N, seg_stride):
    rho_full = np.repeat(rhoK, seg_stride)[:N]
    cost, _ = simulate_cost(rho_full, ds_vec, grade_vec, wind_vec, rider, phys, opt)
    return cost


def plot_results(ds_vec, grade_vec, P_opt, rider_info):
    distance_km = np.cumsum(ds_vec) / 1000.0

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制功率
    color1 = 'tab:blue'
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Power Output (W)', color=color1)
    ax1.plot(distance_km, P_opt, color=color1, label='Optimized Power', linewidth=1.5)
    ax1.axhline(y=rider_info.CP_w, color='r', linestyle='--', label='Critical Power (CP)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 绘制坡度
    ax2 = ax1.twinx()
    color2 = 'tab:gray'
    ax2.set_ylabel('Grade (%)', color=color2)
    ax2.fill_between(distance_km, 0, grade_vec * 100, color=color2, alpha=0.2, label='Course Grade')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f'Optimal Power Distribution vs Grade ({rider_info.rider_type} {rider_info.sex})')
    fig.tight_layout()

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # plt.show()
    plt.savefig('optimized_power_distribution.png', dpi=300)
    plt.close()


def optimize_GRE_DE(csv_path, rider_type, sex, phys, opt):
    df = pd.read_csv(csv_path)
    ds_vec = df["Distance_step_m"].values
    grade_col = "Grade_pct_smooth" if "Grade_pct_smooth" in df.columns else "Grade_pct"
    grade_vec = df[grade_col].values / 100.0
    wind_vec = np.zeros(len(ds_vec))

    rider = build_rider_profile(rider_type, sex)
    N = len(ds_vec)
    K = int(np.ceil(N / opt.seg_stride))
    
    # CMA-ES init
    x0 = 0.5 * np.ones(K)
    sigma0 = 0.2
    
    cma_opts = {
        'bounds': [0.1, 1.0],
        'popsize': opt.popsize,
        'maxiter': opt.maxiter,
        'verbose': 0, # Less output
        'verb_disp': 1,
        'seed': opt.seed if opt.seed is not None else np.random.randint(2**30),
    }

    print(f"开始优化(CMA-ES & Numba): 变量数={K}, 骑手={rider_type}({sex}), seed={cma_opts['seed']}")

    # cma.fmin returns result list/object. res[0] is best_x
    res = cma.fmin(global_objective, x0, sigma0, options=cma_opts,
                   args=(ds_vec, grade_vec, wind_vec, rider, phys, opt, N, opt.seg_stride))
    
    x_final = res[0] # Best solution

    # polish with L-BFGS-B
    lb = 0.1 * np.ones(K)
    ub = 1.0 * np.ones(K)
    res_lbfgs = minimize(global_objective, x_final, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                         args=(ds_vec, grade_vec, wind_vec, rider, phys, opt, N, opt.seg_stride),
                         options={'maxiter': 200, 'ftol': 1e-6})
    x_final = res_lbfgs.x
    print(f"f-value after L-BFGS-B polishing: {res_lbfgs.fun:.4f}")
    
    rho_opt = np.repeat(x_final, opt.seg_stride)[:N]
    cost, diag = simulate_cost(rho_opt, ds_vec, grade_vec, wind_vec, rider, phys, opt)
    Pmax = min(opt.Pmax_abs, rider.Pmax_w)
    P_opt = opt.Pmin + rho_opt * (Pmax - opt.Pmin)

    # 绘图
    plot_results(ds_vec, grade_vec, P_opt, rider)

    return res_lbfgs, diag, P_opt, rider


# =========================
# 5) 执行
# =========================
if __name__ == "__main__":
    CSV_FILE = "./route_with_grade.csv"

    phys_cfg = PhysParams()
    # 将 seed 设为 None 表示每次运行都会生成新的随机初始种群（通常每次结果不同）
    opt_cfg = OptParams(
        seg_stride=20,  # 步长设为100以获得更好的图形细节
        maxiter=1000,  # 迭代数
        popsize=16,
        seed=42  # <-- 设为 None：每次运行随机；设为整数 e.g. 1 则可复现
    )

    try:
        res, diags, p_profile, r_info = optimize_GRE_DE(
            CSV_FILE, "Time trialist", "Men", phys_cfg, opt_cfg
        )

        print("\n" + "=" * 30)
        print("优化完成!")
        print(f"预计耗时: {diags['T_sec'] / 60:.2f} 分钟")
        print(f"标准化功率 (NP): {diags['NP_W']:.1f} W (CP: {r_info.CP_w} W)")
        print(f"终点剩余体力 (W'): {diags['W_end_J'] / 1000:.2f} kJ")
        print("=" * 30)

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{CSV_FILE}'，请检查路径。")