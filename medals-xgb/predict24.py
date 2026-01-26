# ============================================================
# 模型验证：使用2024年以前的数据预测2024年成绩
# ============================================================

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print(">>> 开始训练XGBoost模型预测金银铜牌...")

# 准备训练数据：删除缺失值过多的行
train_df = pd.read_csv("./data/train_df.csv")
test_df = pd.read_csv("./data/test_df.csv")
train_df_clean = train_df.dropna().copy()
print(f">>> 训练数据行数: {len(train_df_clean)} (删除缺失值后)")

# 定义特征列（排除目标变量和标识列）
feature_cols = [
    'Lagged_Squad_Size',
    'Lagged_NOC_Events_Entered',
    'Lagged_Sport_Diversity',
    'Lagged_Global_Total_Events',
    'Lagged_Global_Nations_Count',
    'Lagged_Share_of_Entries',
    'Lagged_Expected_Share',
    'Lagged_RCA_Macro',
    'Lagged_Specialization_Index',
    'Lagged_Total_Medals',
    'Lagged_2_Total_Medals',
    'Lagged_3_Total_Medals',
    'Is_Host',
    'Has_Medal_before'
]

# 检查特征列是否存在
available_features = [col for col in feature_cols if col in train_df_clean.columns]
print(f">>> 可用特征数: {len(available_features)}")

# 准备训练数据
X_train = train_df_clean[available_features].copy()
y_gold = train_df_clean['Gold'].copy()
y_silver = train_df_clean['Silver'].copy()
y_bronze = train_df_clean['Bronze'].copy()

print("=" * 60)
print(">>> 开始模型验证：预测2024年成绩")
print("=" * 60)

# 分离训练数据（2024年以前）和验证数据（2024年）
train_df_before_2024 = train_df[train_df['Year'] < 2024].copy()
train_df_2024 = train_df[train_df['Year'] == 2024].copy()

print(f"\n>>> 训练数据（2024年以前）: {len(train_df_before_2024)} 条")
print(f">>> 验证数据（2024年）: {len(train_df_2024)} 条")

# 准备训练数据：删除缺失值
train_df_clean = train_df_before_2024.dropna().copy()
print(f">>> 清理后的训练数据: {len(train_df_clean)} 条")

# 准备验证数据：填充缺失值
val_df_clean = train_df_2024.copy()
for col in available_features:
    if col in val_df_clean.columns:
        if val_df_clean[col].isna().any():
            median_val = train_df_clean[col].median()
            val_df_clean[col] = val_df_clean[col].fillna(median_val)

# 准备特征和目标变量
X_train_val = train_df_clean[available_features].copy()
X_val_2024 = val_df_clean[available_features].copy()

y_gold_train = train_df_clean['Gold'].copy()
y_silver_train = train_df_clean['Silver'].copy()
y_bronze_train = train_df_clean['Bronze'].copy()

y_gold_actual = val_df_clean['Gold'].copy()
y_silver_actual = val_df_clean['Silver'].copy()
y_bronze_actual = val_df_clean['Bronze'].copy()

print(f"\n>>> 训练集大小: {len(X_train_val)}")
print(f">>> 验证集大小: {len(X_val_2024)}")

# 训练模型并预测2024年
models_val = {}
predictions_val = {}
metrics_val = {}

for target_name, y_train, y_actual in [
    ('Gold', y_gold_train, y_gold_actual),
    ('Silver', y_silver_train, y_silver_actual),
    ('Bronze', y_bronze_train, y_bronze_actual)
]:
    print(f"\n>>> 训练 {target_name} 预测模型（使用2024年以前数据）...")
    
    # 创建XGBoost模型
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    # 训练模型
    model.fit(
        X_train_val, y_train,
        verbose=False
    )
    
    # 预测2024年
    y_pred_2024 = model.predict(X_val_2024)
    y_pred_2024 = np.round(y_pred_2024).astype(int)
    y_pred_2024 = np.clip(y_pred_2024, 0, None)  # 确保非负
    
    # 计算评估指标
    mae = mean_absolute_error(y_actual, y_pred_2024)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_2024))
    
    # 计算准确率（完全匹配的比例）
    exact_match = (y_pred_2024 == y_actual).sum() / len(y_actual)
    
    # 计算误差在±1范围内的比例
    within_1 = (np.abs(y_pred_2024 - y_actual) <= 1).sum() / len(y_actual)
    
    # 计算误差在±2范围内的比例
    within_2 = (np.abs(y_pred_2024 - y_actual) <= 2).sum() / len(y_actual)
    
    print(f">>> {target_name} 模型评估指标:")
    print(f">>>   MAE: {mae:.2f}")
    print(f">>>   RMSE: {rmse:.2f}")
    print(f">>>   完全匹配率: {exact_match*100:.2f}%")
    print(f">>>   误差≤1的比例: {within_1*100:.2f}%")
    print(f">>>   误差≤2的比例: {within_2*100:.2f}%")
    
    models_val[target_name] = model
    predictions_val[target_name] = y_pred_2024
    metrics_val[target_name] = {
        'MAE': mae,
        'RMSE': rmse,
        'Exact_Match': exact_match,
        'Within_1': within_1,
        'Within_2': within_2
    }

# 创建对比结果DataFrame
comparison_df = val_df_clean[['Year', 'NOC']].copy()
comparison_df['Gold_Actual'] = y_gold_actual.values
comparison_df['Gold_Pred'] = predictions_val['Gold']
comparison_df['Gold_Error'] = comparison_df['Gold_Pred'] - comparison_df['Gold_Actual']

comparison_df['Silver_Actual'] = y_silver_actual.values
comparison_df['Silver_Pred'] = predictions_val['Silver']
comparison_df['Silver_Error'] = comparison_df['Silver_Pred'] - comparison_df['Silver_Actual']

comparison_df['Bronze_Actual'] = y_bronze_actual.values
comparison_df['Bronze_Pred'] = predictions_val['Bronze']
comparison_df['Bronze_Error'] = comparison_df['Bronze_Pred'] - comparison_df['Bronze_Actual']

comparison_df['Total_Actual'] = comparison_df['Gold_Actual'] + comparison_df['Silver_Actual'] + comparison_df['Bronze_Actual']
comparison_df['Total_Pred'] = comparison_df['Gold_Pred'] + comparison_df['Silver_Pred'] + comparison_df['Bronze_Pred']
comparison_df['Total_Error'] = comparison_df['Total_Pred'] - comparison_df['Total_Actual']

# 计算总奖牌数的评估指标
total_mae = mean_absolute_error(comparison_df['Total_Actual'], comparison_df['Total_Pred'])
total_rmse = np.sqrt(mean_squared_error(comparison_df['Total_Actual'], comparison_df['Total_Pred']))
total_exact_match = (comparison_df['Total_Pred'] == comparison_df['Total_Actual']).sum() / len(comparison_df)
total_within_1 = (np.abs(comparison_df['Total_Error']) <= 1).sum() / len(comparison_df)
total_within_2 = (np.abs(comparison_df['Total_Error']) <= 2).sum() / len(comparison_df)

print("\n" + "=" * 60)
print(">>> 总奖牌数预测评估:")
print(f">>>   MAE: {total_mae:.2f}")
print(f">>>   RMSE: {total_rmse:.2f}")
print(f">>>   完全匹配率: {total_exact_match*100:.2f}%")
print(f">>>   误差≤1的比例: {total_within_1*100:.2f}%")
print(f">>>   误差≤2的比例: {total_within_2*100:.2f}%")
print("=" * 60)

# 显示预测最准确的前10名
print("\n>>> 预测最准确的前10名（按总奖牌数误差排序）:")
top_accurate = comparison_df.nsmallest(10, 'Total_Error', keep='all')[['NOC', 'Total_Actual', 'Total_Pred', 'Total_Error']]
print(top_accurate.to_string(index=False))

# 显示预测误差最大的前10名
print("\n>>> 预测误差最大的前10名:")
top_error = comparison_df.nlargest(10, 'Total_Error', keep='all')[['NOC', 'Total_Actual', 'Total_Pred', 'Total_Error']]
print(top_error.to_string(index=False))

# 显示实际奖牌数前20名的预测对比
print("\n>>> 实际奖牌数前20名的预测对比:")
top_20_actual = comparison_df.nlargest(20, 'Total_Actual')[['NOC', 'Gold_Actual', 'Gold_Pred', 'Silver_Actual', 'Silver_Pred', 'Bronze_Actual', 'Bronze_Pred', 'Total_Actual', 'Total_Pred', 'Total_Error']]
print(top_20_actual.to_string(index=False))

# 保存对比结果
comparison_df = comparison_df.sort_values(by='Total_Actual', ascending=False)
comparison_df.to_csv("outputs/pred_vs_actual_2024.csv", index=False)
print(f"\n>>> 对比结果已保存到: outputs/pred_vs_actual_2024.csv")

comparison_df.head(20)