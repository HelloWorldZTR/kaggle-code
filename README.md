# Kaggle 入门比赛

## Titanic

- 根据category补充missing data（age）
- 对于categorial column，采用onehot embedding，因为没有大小之分
- random forest的参数设置，如bootstrap在数据不足时，提供更多数据
max_depth和max_leaf_nodes用于在特种值多的时候抑制过拟合

```python
model2 = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    oob_score=True, 
    bootstrap=True,
    max_depth=5,
    # max_leaf_nodes=20
)
```