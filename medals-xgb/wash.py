import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================
# 0. 路径配置
# ============================================================
BASE_DIR = Path("./data")
ATH_PATH = BASE_DIR / "summerOly_athletes.csv"
MEDAL_PATH = BASE_DIR / "summerOly_medal_counts.csv"
HOST_PATH = BASE_DIR / "summerOly_hosts.csv"
PROGRAM_PATH = BASE_DIR / "summerOly_programs.csv"
OUT_TRAIN = BASE_DIR / "train_table_v6_final.csv"


# ============================================================
# 1. 数据加载与清洗函数
# ============================================================
def load_and_clean_data():
    """加载原始数据并进行基础清洗"""
    print(">>> [1/4] 正在读取原始数据...")
    
    # 读取数据
    athletes_df = pd.read_csv(ATH_PATH, encoding="utf-8")
    medals_df = pd.read_csv(MEDAL_PATH, encoding="utf-8")
    hosts_df = pd.read_csv(HOST_PATH, encoding="utf-8")
    programs_df = pd.read_csv(PROGRAM_PATH, encoding="latin1")
    
    # 格式化年份
    athletes_df = athletes_df[athletes_df['Year'] != 1906]  # 排除1906届
    athletes_df["Year"] = athletes_df["Year"].astype(int)
    medals_df["Year"] = medals_df["Year"].astype(int)
    hosts_df["Year"] = hosts_df["Year"].astype(int)
    
    return athletes_df, medals_df, hosts_df, programs_df


def clean_noc_and_mapping(athletes_df, medals_df):
    """清洗NOC代码并构建映射关系"""
    print(">>> [2/4] 清洗 NOC 与构建映射...")
    
    # 清洗无效NOC
    invalid_nocs = ['AIN', 'EOR', 'ROT', 'IOA', 'IOP', 'ZZX', 'MIX', 'UNK']
    athletes_cleaned = athletes_df[~athletes_df['NOC'].isin(invalid_nocs)].copy()
    athletes_cleaned.loc[athletes_cleaned['NOC'] == 'ROC', 'NOC'] = 'RUS'
    
    # 过滤体育项目
    if 'Sport' in athletes_cleaned.columns:
        athletes_cleaned = athletes_cleaned[athletes_cleaned['Sport'] != 'Art Competitions']
    if 'Season' in athletes_cleaned.columns:
        athletes_cleaned = athletes_cleaned[athletes_cleaned['Season'] == 'Summer']
    
    # 建立NOC映射
    name_to_code_map = athletes_cleaned.set_index('Team')['NOC'].to_dict()
    manual_patches = {
        "Soviet Union": "URS", "East Germany": "GDR", "West Germany": "FRG",
        "Great Britain": "GBR", "United States": "USA", "China": "CHN",
        "People's Republic of China": "CHN", "ROC": "RUS", "Russia": "RUS",
        "Unified Team": "EUN", "Korea, South": "KOR", "South Korea": "KOR"
    }
    name_to_code_map.update(manual_patches)
    
    # 清洗奖牌数据
    medals_cleaned = medals_df.copy()
    medals_cleaned['NOC_Code'] = medals_cleaned['NOC'].map(name_to_code_map).fillna(medals_cleaned['NOC'])
    medals_cleaned.loc[medals_cleaned['NOC_Code'] == 'ROC', 'NOC_Code'] = 'RUS'
    medals_cleaned = medals_cleaned[~medals_cleaned['NOC_Code'].isin(invalid_nocs)]
    
    return athletes_cleaned, medals_cleaned, name_to_code_map


# ============================================================
# 执行数据加载与清洗
# ============================================================
ath, med, hst, prog = load_and_clean_data()
ath, med, name_to_code_map = clean_noc_and_mapping(ath, med)

# ============================================================
# 2. 构建基础特征函数
# ============================================================
def build_basic_features(athletes_df, medals_df, programs_df):
    """构建宏观和微观基础特征"""
    print(">>> [3/4] 构建特征工程 (宏观+微观)...")
    
    # --- A. 宏观特征 ---
    year_cols = [c for c in programs_df.columns if c.strip().isdigit()]
    prog_numeric = programs_df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    macro_features = pd.DataFrame()
    macro_features["Year"] = [int(y) for y in year_cols]
    macro_features["Global_Total_Events"] = prog_numeric.sum().values
    macro_features = macro_features.merge(
        athletes_df.groupby("Year")["NOC"].nunique().reset_index(name="Global_Nations_Count"),
        on="Year", how="left"
    )
    # 只要2024及以前
    macro_features = macro_features[macro_features["Year"] <= 2024]
    
    # --- B. 微观特征 ---
    # 1. 队伍规模 (多少人)
    squad_size = athletes_df.groupby(["Year", "NOC"])["Name"].nunique().reset_index(name="Squad_Size")
    # 2. 参赛项目数 (参加了多少个小项)
    noc_events = athletes_df.groupby(["Year", "NOC"])["Event"].nunique().reset_index(name="NOC_Events_Entered")
    # 3. 体育多样性 (参加了多少个大项)
    sport_diversity = athletes_df.groupby(["Year", "NOC"])["Sport"].nunique().reset_index(name="Sport_Diversity")
    
    # 合并微观特征
    micro_features = squad_size.merge(noc_events, on=["Year", "NOC"], how="left")
    micro_features = micro_features.merge(sport_diversity, on=["Year", "NOC"], how="left")
    
    # --- C. 合并奖牌数据 ---
    target_cols = ["Year", "NOC_Code", "Total"]
    features_with_medals = micro_features.merge(
        medals_df[target_cols], 
        left_on=["Year", "NOC"], 
        right_on=["Year", "NOC_Code"], 
        how="left"
    )
    features_with_medals["Total"] = features_with_medals["Total"].fillna(0).astype(int)
    features_with_medals = features_with_medals.drop(columns=['NOC_Code'])
    
    # --- D. 合并宏观特征并添加时序特征 ---
    basic_features_df = features_with_medals.merge(macro_features, on="Year", how="left")
    basic_features_df = basic_features_df.sort_values(["NOC", "Year"])
    
    # 历史累计表现
    basic_features_df["Historical_Total"] = basic_features_df.groupby("NOC", group_keys=False)["Total"].apply(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    
    return basic_features_df

basic_features_df = build_basic_features(ath, med, prog)

# ============================================================
# 3. 构建特征 (宏观 + 微观)
# ============================================================


def build_features(ath, med, prog):
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

    # 2. 历史累计表现 (使用 group_keys=False 修复报错)
    master["Historical_Total"] = master.groupby("NOC", group_keys=False)["Total"].apply(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)


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

    # 曾经是否得过奖
    final_df["Has_Medal_before"] = (final_df["Historical_Total"] > 0).astype(int)


    return final_df

fe_df = build_features(ath, med, prog)


def lag_features(df, host_dict=None):
    """创建滞后特征，并为2028年创建预测数据（使用2024年的滞后特征）"""
    df = df.copy()
    # 确保按 NOC 和 Year 排序
    df = df.sort_values(["NOC", "Year"]).reset_index(drop=True)
    
    # 需要滞后的列
    need_lag_cols = [
        "Squad_Size",
        "NOC_Events_Entered",
        "Sport_Diversity",
        "Global_Total_Events",
        "Global_Nations_Count",
        "Share_of_Entries",
        "Expected_Share",
        "RCA_Macro",
        "Specialization_Index",
    ]
    
    # 分离训练数据（2024及以前）
    train_data = df[df["Year"] <= 2024].copy()
    
    # 为训练数据创建滞后特征
    res_df = pd.DataFrame()
    res_df["Year"] = train_data["Year"]
    res_df["NOC"] = train_data["NOC"]
    
    # 创建滞后特征：使用上一年的数据
    for col in need_lag_cols:
        res_df[f"Lagged_{col}"] = train_data.groupby("NOC")[col].shift(1)
    
    # 创建奖牌的滞后特征
    res_df["Lagged_Total_Medals"] = train_data.groupby("NOC")["Total"].shift(1)
    res_df["Lagged_2_Total_Medals"] = train_data.groupby("NOC")["Total"].shift(2)
    res_df["Lagged_3_Total_Medals"] = train_data.groupby("NOC")["Total"].shift(3)
    
    # 复制非滞后特征
    res_df["Is_Host"] = train_data["Is_Host"]
    res_df["Has_Medal_before"] = train_data["Has_Medal_before"]
    res_df["Total"] = train_data["Total"]  # 保留目标变量用于训练
    
    # ============================================================
    # 为2028年创建预测数据（使用2024年的特征作为滞后特征）
    # ============================================================
    print(">>> 为2028年创建预测数据，使用2024年的特征作为滞后特征...")
    
    # 获取2024年的所有NOC数据
    df_2024 = train_data[train_data["Year"] == 2024].copy()
    
    if len(df_2024) == 0:
        print(">>> 警告：未找到2024年数据，无法创建2028年预测数据")
        df_2028 = pd.DataFrame()
        return res_df, df_2028
    
    # 为每个在2024年有数据的NOC创建2028年记录
    df_2028_lagged = pd.DataFrame()
    df_2028_lagged["Year"] = [2028] * len(df_2024)
    df_2028_lagged["NOC"] = df_2024["NOC"].values
    
    # 使用2024年的特征值作为滞后特征
    for col in need_lag_cols:
        df_2028_lagged[f"Lagged_{col}"] = df_2024[col].values
    
    # 创建奖牌的滞后特征（使用历史数据）
    lagged_medals_1 = []
    lagged_medals_2 = []
    lagged_medals_3 = []
    
    for noc in df_2024["NOC"]:
        noc_history = train_data[train_data["NOC"] == noc].sort_values("Year", ascending=False)
        if len(noc_history) >= 1:
            lagged_medals_1.append(noc_history["Total"].iloc[0])  # 2024年
        else:
            lagged_medals_1.append(0)
        
        if len(noc_history) >= 2:
            lagged_medals_2.append(noc_history["Total"].iloc[1])  # 2020年
        else:
            lagged_medals_2.append(0)
        
        if len(noc_history) >= 3:
            lagged_medals_3.append(noc_history["Total"].iloc[2])  # 2016年
        else:
            lagged_medals_3.append(0)
    
    df_2028_lagged["Lagged_Total_Medals"] = lagged_medals_1
    df_2028_lagged["Lagged_2_Total_Medals"] = lagged_medals_2
    df_2028_lagged["Lagged_3_Total_Medals"] = lagged_medals_3
    
    # 处理Has_Medal_before特征（基于2024年及之前的历史累计）
    df_2028_lagged["Has_Medal_before"] = df_2024["Has_Medal_before"].values
    
    # 处理Is_Host特征（2028年是洛杉矶奥运会，USA是东道主）
    if host_dict is not None:
        df_2028_lagged["Is_Host"] = df_2028_lagged["NOC"].apply(
            lambda noc: 1 if host_dict.get(2028) == noc else 0
        )
    else:
        # 如果没有提供host_dict，默认USA是2028年东道主
        df_2028_lagged["Is_Host"] = (df_2028_lagged["NOC"] == "USA").astype(int)
    
    return res_df, df_2028_lagged

train_df, test_df = lag_features(fe_df)

# ============================================================
# 合并金银铜数据到train_df
# ============================================================
print("\n>>> 合并金银铜数据到训练数据...")

# 从med数据中提取金银铜列，使用NOC_Code和Year对齐
medal_cols = med[["NOC_Code", "Year", "Gold", "Silver", "Bronze"]].copy()
medal_cols = medal_cols.rename(columns={"NOC_Code": "NOC"})

# 合并到train_df，对齐NOC和Year
train_df = train_df.merge(
    medal_cols,
    on=["NOC", "Year"],
    how="left"
)

# 填充缺失值（某些NOC在某个年份可能没有奖牌数据）
train_df["Gold"] = train_df["Gold"].fillna(0).astype(int)
train_df["Silver"] = train_df["Silver"].fillna(0).astype(int)
train_df["Bronze"] = train_df["Bronze"].fillna(0).astype(int)


train_df.to_csv("./data/train_df.csv", index=False)
test_df.to_csv("./data/test_df.csv", index=False)