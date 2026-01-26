# ============================================================
# 使用XGBoost预测2028年金银铜牌数量
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

# 准备测试数据（2028年）
# 检查test_df是否有缺失值，如果有则填充
test_df_clean = test_df.copy()
for col in available_features:
    if col in test_df_clean.columns:
        if test_df_clean[col].isna().any():
            # 使用中位数填充
            median_val = train_df_clean[col].median()
            test_df_clean[col] = test_df_clean[col].fillna(median_val)
            print(f">>> 填充 {col} 的缺失值，使用中位数: {median_val}")

X_test = test_df_clean[available_features].copy()

print(f"\n>>> 训练集大小: {len(X_train)}")
print(f">>> 测试集大小: {len(X_test)}")

# 训练三个模型分别预测金银铜
models = {}
predictions = {}

for target_name, y_target in [('Gold', y_gold), ('Silver', y_silver), ('Bronze', y_bronze)]:
    print(f"\n>>> 训练 {target_name} 预测模型...")
    
    # 分割训练集和验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_target, test_size=0.2, random_state=42
    )
    
    # 创建XGBoost模型
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    # 训练模型
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 验证集预测
    y_pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    print(f">>> {target_name} 模型 - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # 保存模型
    models[target_name] = model
    
    # 对2028年数据进行预测
    pred_2028 = model.predict(X_test)
    predictions[target_name] = pred_2028

# 创建预测结果DataFrame
pred_df_2028 = test_df_clean[['Year', 'NOC']].copy()
pred_df_2028['Gold'] = np.round(predictions['Gold']).astype(int)
pred_df_2028['Silver'] = np.round(predictions['Silver']).astype(int)
pred_df_2028['Bronze'] = np.round(predictions['Bronze']).astype(int)
pred_df_2028['Total'] = pred_df_2028['Gold'] + pred_df_2028['Silver'] + pred_df_2028['Bronze']

# 确保预测值非负
pred_df_2028['Gold'] = pred_df_2028['Gold'].clip(lower=0)
pred_df_2028['Silver'] = pred_df_2028['Silver'].clip(lower=0)
pred_df_2028['Bronze'] = pred_df_2028['Bronze'].clip(lower=0)
pred_df_2028['Total'] = pred_df_2028['Total'].clip(lower=0)

print("\n>>> 2028年预测结果:")
print(f">>> 预测国家数: {len(pred_df_2028)}")
print(f">>> 预计获得奖牌的国家数: {(pred_df_2028['Total'] > 0).sum()}")
print(f"\n>>> 2028年预测奖牌总数:")
print(f">>> 金牌: {pred_df_2028['Gold'].sum()}")
print(f">>> 银牌: {pred_df_2028['Silver'].sum()}")
print(f">>> 铜牌: {pred_df_2028['Bronze'].sum()}")
print(f">>> 总计: {pred_df_2028['Total'].sum()}")

# 显示前20名预测结果
print("\n>>> 2028年预测奖牌数前20名:")
top_20 = pred_df_2028.nlargest(20, 'Total')[['NOC', 'Gold', 'Silver', 'Bronze', 'Total']]
print(top_20.to_string(index=False))

# 保存预测结果
pred_df_2028 = pred_df_2028.sort_values(by='Total', ascending=False)
pred_df_2028.to_csv("outputs/lstm_pred_2028.csv", index=False)
print(f"\n>>> 预测结果已保存到: outputs/lstm_pred_2028.csv")

pred_df_2028.head(20)