import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================
# 0. 路径配置 (请务必修改为您电脑上的实际路径)
# ============================================================
BASE_DIR = Path(__file__).parent / "data"
ATH_PATH     = BASE_DIR / "summerOly_athletes.csv"
MEDAL_PATH   = BASE_DIR / "summerOly_medal_counts.csv"
HOST_PATH    = BASE_DIR / "summerOly_hosts.csv"
PROGRAM_PATH = BASE_DIR / "summerOly_programs.csv"
OUT_TRAIN    = BASE_DIR / "train_table_v6_final.csv"

# ============================================================
# 1. 读取数据
# ============================================================
print(">>> [1/4] 正在读取原始数据...")
ath = pd.read_csv(ATH_PATH, encoding="utf-8")
med = pd.read_csv(MEDAL_PATH, encoding="utf-8")
hst = pd.read_csv(HOST_PATH, encoding="utf-8")
# programs 可能有编码问题，尝试两种编码
prog = pd.read_csv(PROGRAM_PATH, encoding="latin1") 


# 格式化年份
ath = ath[ath['Year'] != 1906] # 排除1906届
ath["Year"] = ath["Year"].astype(int)
med["Year"] = med["Year"].astype(int)
hst["Year"] = hst["Year"].astype(int)

# ============================================================
# 2. 基础清洗与映射
# ============================================================
print(">>> [2/4] 清洗 NOC 与构建映射...")
invalid_nocs = ['AIN', 'EOR', 'ROT', 'IOA', 'IOP', 'ZZX', 'MIX', 'UNK']
ath = ath[~ath['NOC'].isin(invalid_nocs)]
ath.loc[ath['NOC'] == 'ROC', 'NOC'] = 'RUS'

if 'Sport' in ath.columns: ath = ath[ath['Sport'] != 'Art Competitions']
if 'Season' in ath.columns: ath = ath[ath['Season'] == 'Summer']

# 建立 NOC 映射
name_to_code_map = ath.set_index('Team')['NOC'].to_dict()
manual_patches = {
    "Soviet Union": "URS", "East Germany": "GDR", "West Germany": "FRG",
    "Great Britain": "GBR", "United States": "USA", "China": "CHN",
    "People's Republic of China": "CHN", "ROC": "RUS", "Russia": "RUS",
    "Unified Team": "EUN", "Korea, South": "KOR", "South Korea": "KOR"
}
name_to_code_map.update(manual_patches)

med['NOC_Code'] = med['NOC'].map(name_to_code_map).fillna(med['NOC'])
med.loc[med['NOC_Code'] == 'ROC', 'NOC_Code'] = 'RUS'
med = med[~med['NOC_Code'].isin(invalid_nocs)]

# ============================================================
# 3. 构建特征 (宏观 + 微观)
# ============================================================
print(">>> [3/4] 构建特征工程 (宏观+微观)...")

# --- A. 宏观特征 ---
year_cols = [c for c in prog.columns if c.strip().isdigit()]
prog_numeric = prog[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

macro_stats = pd.DataFrame()
macro_stats["Year"] = [int(y) for y in year_cols]
macro_stats["Global_Total_Events"] = prog_numeric.sum().values
macro_stats = macro_stats.merge(
    ath.groupby("Year")["NOC"].nunique().reset_index(name="Global_Nations_Count"),
    on="Year", how="left"
)
# 只要2024及以前
macro_stats = macro_stats[macro_stats["Year"] <= 2024]

# --- B. 微观特征 (【关键修改点】：加入了 Sport_Diversity) ---
# 1. 队伍规模 (多少人)
squad = ath.groupby(["Year", "NOC"])["Name"].nunique().reset_index(name="Squad_Size")
# 2. 参赛项目数 (参加了多少个小项)
noc_events = ath.groupby(["Year", "NOC"])["Event"].nunique().reset_index(name="NOC_Events_Entered")
# 3. 体育多样性 (参加了多少个大项) -> 也就是 "Specialization Index" 的分母
sport_div = ath.groupby(["Year", "NOC"])["Sport"].nunique().reset_index(name="Sport_Diversity")

# 合并微观特征
features = squad.merge(noc_events, on=["Year", "NOC"], how="left")
features = features.merge(sport_div, on=["Year", "NOC"], how="left")

# --- C. 合并奖牌 ---
target_cols = ["Year", "NOC_Code", "Total"]
master = features.merge(med[target_cols], left_on=["Year", "NOC"], right_on=["Year", "NOC_Code"], how="left")
master["Total"] = master["Total"].fillna(0).astype(int)
master = master.drop(columns=['NOC_Code'])

# --- D. 关键时序特征 ---
master = master.sort_values(["NOC", "Year"])

# 1. 上一届表现
master["Lagged_Total_Medals"] = master.groupby("NOC")["Total"].shift(1)

# 2. 历史累计表现 (使用 group_keys=False 修复报错)
master["Historical_Total"] = master.groupby("NOC", group_keys=False)["Total"].apply(
    lambda x: x.cumsum().shift(1)
).fillna(0)

# 3. 破蛋候选人标记
master["Is_First_Time_Candidate"] = (master["Historical_Total"] == 0).astype(int)

# ============================================================
# 4. 特征增强 (RCA & Specialization) 与 保存
# ============================================================
print(">>> [4/4] 计算高级特征 (RCA & 专注度) 并保存...")
final_df = master.merge(macro_stats, on="Year", how="left")

# --- 【新增核心逻辑 Start】 ---

# 1. 计算宏观参与 RCA (RCA_Macro)
# 分子：该国在当年的项目份额 (Share of Entries)
yearly_total_entries = final_df.groupby("Year")["NOC_Events_Entered"].transform("sum")
final_df["Share_of_Entries"] = final_df["NOC_Events_Entered"] / yearly_total_entries

# 分母：期望份额 (Expected Share = 1 / Global Nations Count)
final_df["Expected_Share"] = 1 / final_df["Global_Nations_Count"]

# RCA 计算
final_df["RCA_Macro"] = final_df["Share_of_Entries"] / final_df["Expected_Share"]
# 填充可能产生的 inf (虽然不太可能，以防万一)
final_df["RCA_Macro"] = final_df["RCA_Macro"].replace([np.inf, -np.inf], 0).fillna(0)

# 2. 计算专注度指数 (Specialization Index)
# 公式：Events_Entered / Sport_Diversity
# 含义：平均每个大项报了多少个小项。数值高说明“兵力集中”。
# 防止除以0
final_df["Sport_Diversity"] = final_df["Sport_Diversity"].replace(0, 1)
final_df["Specialization_Index"] = final_df["NOC_Events_Entered"] / final_df["Sport_Diversity"]

# --- 【新增核心逻辑 End】 ---

# 东道主处理
hst["HostCountry"] = hst["Host"].apply(lambda x: x.split(",")[-1].strip())
hst["HostNOC"] = hst["HostCountry"].map(name_to_code_map)
hst_map = {
    "United Kingdom": "GBR", "Australia": "AUS", "Japan": "JPN",
    "China": "CHN", "Brazil": "BRA", "Greece": "GRE"
}
hst["HostNOC"] = hst["HostNOC"].fillna(hst["HostCountry"].map(hst_map))
host_dict = hst.set_index("Year")["HostNOC"].to_dict()

# 计算 Host 特征
final_df["Is_Host"] = final_df.apply(lambda r: 1 if host_dict.get(r["Year"]) == r["NOC"] else 0, axis=1)
final_df["Is_Next_Host"] = final_df.apply(lambda r: 1 if host_dict.get(r["Year"] + 4) == r["NOC"] else 0, axis=1)

# 去除第一年没有滞后数据的行 (Train Data)
train_df = final_df.dropna(subset=["Lagged_Total_Medals"]).copy()

# 顺便把 Total 转换为 Has_Medal，方便后续直接用
train_df["Has_Medal"] = (train_df["Total"] > 0).astype(int)

# 保存
train_df.to_csv(OUT_TRAIN, index=False)
print("="*60)
print(f"SUCCESS! 数据处理完成。")
print(f"包含 RCA 和 Specialization_Index 的文件已保存至: {OUT_TRAIN}")
print("现在请运行预测脚本 (Random Forest)，它将自动读取这些新特征。")
print("="*60)