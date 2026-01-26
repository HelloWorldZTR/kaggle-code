"""LSTM-based medal count forecasting using pre-extracted features.

This script:
- Loads pre-extracted features from data.csv
- Builds sliding-window sequences (per country) and trains an LSTM to predict next-Games gold, silver, and bronze medal counts
- Uses weighted Huber loss to better handle extreme values
- Trains on all years except the last four Games, then evaluates predictions for the last four
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


DATA_CSV = os.path.join(os.path.dirname(__file__), "data.csv")


class WeightedHuberLoss(torch.nn.Module):
	"""Huber Loss with different weights for gold, silver, bronze medals.
	
	Huber Loss is more sensitive to outliers than MSE while being smoother than MAE.
	Weights: gold > silver > bronze to emphasize the importance of gold medals.
	"""
	def __init__(self, medal_weights: Tuple[float, float, float] = (3.0, 2.0, 1.0), delta: float = 1.0):
		super().__init__()
		self.medal_weights = torch.tensor(medal_weights, dtype=torch.float32)
		self.delta = delta

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			pred: (batch_size, 3) predictions [gold, silver, bronze]
			target: (batch_size, 3) targets [gold, silver, bronze]
		Returns:
			Weighted Huber loss
		"""
		device = pred.device
		weights = self.medal_weights.to(device)
		
		# Calculate Huber loss for each medal type
		error = pred - target
		abs_error = torch.abs(error)
		
		# Huber loss: quadratic for small errors, linear for large errors
		huber_loss = torch.where(
			abs_error <= self.delta,
			0.5 * error ** 2,
			self.delta * (abs_error - 0.5 * self.delta)
		)
		
		# Apply weights: gold gets highest weight
		weighted_loss = huber_loss * weights.unsqueeze(0)
		
		# Give higher weight to samples with larger medal counts (to emphasize extremes)
		# Use a more aggressive weighting: 1.0 + 0.2 * total_medals + 0.01 * total_medals^2
		# This quadratically increases weight for high medal counts
		total_medals = target.sum(dim=1, keepdim=True)
		sample_weights = 1.0 + 0.2 * total_medals + 0.01 * (total_medals ** 2)
		weighted_loss = weighted_loss * sample_weights
		
		return weighted_loss.mean()


class TorchLSTM(torch.nn.Module):
	def __init__(self, time_steps: int, n_features: int, n_targets: int = 3):
		super().__init__()
		self.lstm1 = torch.nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
		self.dropout = torch.nn.Dropout(0.2)
		self.lstm2 = torch.nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
		self.head = torch.nn.Linear(32, n_targets)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x, _ = self.lstm1(x)
		x = self.dropout(x)
		x, _ = self.lstm2(x)
		x = x[:, -1, :]
		return self.head(x)


def build_sequences(
	df: pd.DataFrame,
	time_steps: int,
	feature_cols: List[str],
	target_cols: List[str],
	test_years: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int]]]:
	"""Build sliding window sequences for LSTM training.
	
	Args:
		df: DataFrame with features and targets
		time_steps: Number of time steps in the sequence
		feature_cols: List of feature column names
		target_cols: List of target column names
		test_years: List of years to use as test set
	Returns:
		X_train, y_train, X_test, y_test, test_meta
	"""
	X_train: List[np.ndarray] = []
	y_train: List[np.ndarray] = []
	X_test: List[np.ndarray] = []
	y_test: List[np.ndarray] = []
	test_meta: List[Tuple[str, int]] = []

	for noc, g in df.groupby("NOC"):
		g = g.sort_values("Year")
		years = g["Year"].tolist()
		feats = g[feature_cols].to_numpy(dtype=float)
		targets = g[target_cols].to_numpy(dtype=float)

		for i in range(len(years) - time_steps):
			target_year = years[i + time_steps]
			window = feats[i : i + time_steps]
			target = targets[i + time_steps]
			if target_year in test_years:
				X_test.append(window)
				y_test.append(target)
				test_meta.append((noc, target_year))
			else:
				X_train.append(window)
				y_train.append(target)

	return (
		np.array(X_train, dtype=float),
		np.array(y_train, dtype=float),
		np.array(X_test, dtype=float),
		np.array(y_test, dtype=float),
		test_meta,
	)


def main() -> None:
	time_steps = 3

	# Load data
	df = pd.read_csv(DATA_CSV)
	
	# Identify feature columns (exclude Year, NOC, and target columns)
	target_cols = ["Gold", "Silver", "Bronze"]
	exclude_cols = ["Year", "NOC", "Gold", "Silver", "Bronze", "Total"]
	feature_cols = [col for col in df.columns if col not in exclude_cols]
	
	print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
	print(f"Target columns: {target_cols}")
	
	# Get unique years and identify test years (last 4 unique years)
	unique_years = sorted(df["Year"].unique())
	if len(unique_years) < time_steps + 4:
		raise ValueError("Not enough years to create train/test splits.")
	test_years = unique_years[-4:]
	
	print(f"All years: {unique_years}")
	print(f"Test years (last 4): {test_years}")
	
	# Build sequences
	X_train, y_train, X_test, y_test, test_meta = build_sequences(
		df, time_steps, feature_cols, target_cols, test_years
	)

	if X_train.size == 0 or X_test.size == 0:
		raise ValueError("No train/test samples were created; check data alignment.")

	print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

	# Scale features across all time steps
	scaler = StandardScaler()
	n_features = len(feature_cols)
	X_train_flat = X_train.reshape(-1, n_features)
	X_test_flat = X_test.reshape(-1, n_features)
	scaler.fit(X_train_flat)
	X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
	X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

	# Setup model and training
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	print(f"Using device: {device}")
	
	model = TorchLSTM(time_steps, n_features, n_targets=3).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	# Use weighted Huber loss: gold (3.0) > silver (2.0) > bronze (1.0)
	# Huber loss is more sensitive to outliers than MSE
	loss_fn = WeightedHuberLoss(medal_weights=(3.0, 2.0, 1.0), delta=1.0).to(device)
	epochs = 100

	X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
	y_train_t = torch.tensor(y_train, dtype=torch.float32)
	train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

	X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

	# Training loop
	print("\nTraining...")
	for epoch in range(epochs):
		model.train()
		epoch_loss = 0.0
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			pred = model(xb)
			loss = loss_fn(pred, yb)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item() * xb.size(0)
		epoch_loss /= len(train_loader.dataset)
		if (epoch + 1) % 10 == 0:
			print(f"Epoch {epoch+1:03d} | train loss {epoch_loss:.4f}")

	# Evaluation
	model.eval()
	with torch.no_grad():
		preds = model(X_test_t).cpu().numpy()
	
	# Calculate MAE for each medal type
	mae_gold = mean_absolute_error(y_test[:, 0], preds[:, 0])
	mae_silver = mean_absolute_error(y_test[:, 1], preds[:, 1])
	mae_bronze = mean_absolute_error(y_test[:, 2], preds[:, 2])
	mae_total = (mae_gold + mae_silver + mae_bronze) / 3

	# Create results DataFrame
	results = pd.DataFrame(
		{
			"NOC": [noc for noc, _ in test_meta],
			"Year": [year for _, year in test_meta],
			"actual_gold": y_test[:, 0],
			"pred_gold": preds[:, 0],
			"actual_silver": y_test[:, 1],
			"pred_silver": preds[:, 1],
			"actual_bronze": y_test[:, 2],
			"pred_bronze": preds[:, 2],
		}
	)

	results = results.sort_values(["Year", "NOC"])
	os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
	out_path = os.path.join(os.path.dirname(__file__), "outputs", "lstm_pred_vs_actual.csv")
	results.to_csv(out_path, index=False)

	print(f"\n{'='*60}")
	print(f"Test years: {test_years}")
	print(f"Test MAE - Gold: {mae_gold:.3f}, Silver: {mae_silver:.3f}, Bronze: {mae_bronze:.3f}")
	print(f"Test MAE (average): {mae_total:.3f}")
	print(f"Saved predictions to {out_path}")
	print(f"\nTop 10 predictions by actual gold medals:")
	print(results.nlargest(10, "actual_gold")[["NOC", "Year", "actual_gold", "pred_gold", "actual_silver", "pred_silver", "actual_bronze", "pred_bronze"]])


if __name__ == "__main__":
	main()
