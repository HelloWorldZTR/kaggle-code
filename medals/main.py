"""LSTM-based medal count forecasting using Olympic datasets.

This script:
- Loads medal counts, host info, event counts, and athlete-level data.
- Aggregates athlete data into country-year features (squad size, medalists, diversity, etc.).
- Builds sliding-window sequences (per country) and trains an LSTM to predict next-Games gold, silver, and bronze medal counts.
- Trains on all years except the last four Games, then evaluates predictions for the last four.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MEDAL_COUNTS_CSV = os.path.join(DATA_DIR, "summerOly_medal_counts.csv")
HOSTS_CSV = os.path.join(DATA_DIR, "summerOly_hosts.csv")
PROGRAMS_CSV = os.path.join(DATA_DIR, "summerOly_programs.csv")
ATHLETES_CSV = os.path.join(DATA_DIR, "summerOly_athletes.csv")


def normalize_country(name: str) -> str:
	"""Lowercase and keep letters/spaces only for fuzzy matching."""
	cleaned = re.sub(r"[^a-zA-Z\s]", "", name or "").lower()
	return re.sub(r"\s+", " ", cleaned).strip()


def load_medal_counts(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = df.rename(columns={"NOC": "team", "Gold": "gold", "Silver": "silver", "Bronze": "bronze", "Total": "total", "Year": "year"})
	df["year"] = df["year"].astype(int)
	return df[["team", "year", "gold", "silver", "bronze", "total"]]


def load_hosts(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)

	def extract_country(cell: str) -> str:
		cell = (cell or "").replace("\xa0", " ")
		# Drop canceled rows quickly.
		if "Cancelled" in cell:
			return ""
		parts = [p.strip() for p in cell.split(",") if p.strip()]
		return parts[-1] if parts else ""

	df["host_country"] = df["Host"].apply(extract_country)
	df = df[["Year", "host_country"]].rename(columns={"Year": "year"})
	df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
	df = df.dropna(subset=["year"]).astype({"year": int})
	return df


def load_program_events(path: str) -> pd.DataFrame:
	# summerOly_programs.csv contains non-UTF-8 chars; latin1 avoids decode errors.
	df = pd.read_csv(path, encoding="latin1")
	total_row = df[df["Sport"].str.contains("Total events", case=False, na=False)]
	if total_row.empty:
		return pd.DataFrame(columns=["year", "total_events"])

	# Columns that look like years.
	year_cols = [c for c in total_row.columns if re.fullmatch(r"\d{4}", str(c))]
	melted = total_row[year_cols].T.reset_index()
	melted.columns = ["year", "total_events"]
	melted["year"] = melted["year"].astype(int)
	melted["total_events"] = pd.to_numeric(melted["total_events"], errors="coerce")
	return melted


def build_athlete_features(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = df.rename(columns={"Team": "team", "Year": "year", "Medal": "medal"})
	df["year"] = df["year"].astype(int)

	grouped = df.groupby(["team", "year"])
	base = grouped.agg(
		athlete_count=("Name", "nunique"),
		medalist_count=("medal", lambda x: (x != "No medal").sum()),
		sport_count=("Sport", "nunique"),
		event_count=("Event", "nunique"),
	).reset_index()
	base["conversion_rate"] = base["medalist_count"] / base["athlete_count"].replace(0, np.nan)
	base["conversion_rate"] = base["conversion_rate"].fillna(0.0)

	# Returning athlete ratio vs. previous Games for the same team.
	name_sets: Dict[Tuple[str, int], set] = grouped["Name"].apply(set).to_dict()
	returning: List[float] = []
	for _, row in base.iterrows():
		team, year = row["team"], row["year"]
		prev_years = sorted({y for (t, y) in name_sets if t == team and y < year})
		if not prev_years:
			returning.append(0.0)
			continue
		prev_year = prev_years[-1]
		prev_names = name_sets[(team, prev_year)]
		cur_names = name_sets[(team, year)]
		overlap = len(prev_names & cur_names)
		ratio = overlap / row["athlete_count"] if row["athlete_count"] else 0.0
		returning.append(ratio)
	base["returning_ratio"] = returning
	return base


def merge_features() -> pd.DataFrame:
	medals = load_medal_counts(MEDAL_COUNTS_CSV)
	hosts = load_hosts(HOSTS_CSV)
	events = load_program_events(PROGRAMS_CSV)
	athlete_feats = build_athlete_features(ATHLETES_CSV)

	df = medals.merge(athlete_feats, on=["team", "year"], how="left")
	df = df.merge(events, on="year", how="left")
	df = df.merge(hosts, on="year", how="left")

	df["is_host"] = df.apply(
		lambda r: int(bool(r["host_country"]) and normalize_country(r["host_country"]) == normalize_country(r["team"])),
		axis=1,
	)

	df = df.drop(columns=["host_country"], errors="ignore")
	df = df.fillna(0)
	return df


def build_sequences(
	df: pd.DataFrame,
	time_steps: int,
	feature_cols: List[str],
	target_cols: List[str],
	test_years: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int]]]:
	X_train: List[np.ndarray] = []
	y_train: List[np.ndarray] = []
	X_test: List[np.ndarray] = []
	y_test: List[np.ndarray] = []
	test_meta: List[Tuple[str, int]] = []

	for team, g in df.groupby("team"):
		g = g.sort_values("year")
		years = g["year"].tolist()
		feats = g[feature_cols].to_numpy(dtype=float)
		targets = g[target_cols].to_numpy(dtype=float)

		for i in range(len(years) - time_steps):
			target_year = years[i + time_steps]
			window = feats[i : i + time_steps]
			target = targets[i + time_steps]
			if target_year in test_years:
				X_test.append(window)
				y_test.append(target)
				test_meta.append((team, target_year))
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


def main() -> None:
	time_steps = 3

	df = merge_features()
	unique_years = sorted(df["year"].unique())
	if len(unique_years) < time_steps + 4:
		raise ValueError("Not enough years to create train/test splits.")
	test_years = unique_years[-4:]

	feature_cols = [
		"gold",
		"silver",
		"bronze",
		"total",
		"athlete_count",
		"medalist_count",
		"conversion_rate",
		"sport_count",
		"event_count",
		"returning_ratio",
		"total_events",
		"is_host",
	]
	target_cols = ["gold", "silver", "bronze"]

	X_train, y_train, X_test, y_test, test_meta = build_sequences(
		df, time_steps, feature_cols, target_cols, test_years
	)

	if X_train.size == 0 or X_test.size == 0:
		raise ValueError("No train/test samples were created; check data alignment.")

	# Scale features across all time steps.
	scaler = StandardScaler()
	n_features = len(feature_cols)
	X_train_flat = X_train.reshape(-1, n_features)
	X_test_flat = X_test.reshape(-1, n_features)
	scaler.fit(X_train_flat)
	X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
	X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
		if (epoch + 1) % 5 == 0:
			print(f"Epoch {epoch+1:02d} | train loss {epoch_loss:.4f}")

	model.eval()
	with torch.no_grad():
		preds = model(X_test_t).cpu().numpy()
	
	# Calculate MAE for each medal type
	mae_gold = mean_absolute_error(y_test[:, 0], preds[:, 0])
	mae_silver = mean_absolute_error(y_test[:, 1], preds[:, 1])
	mae_bronze = mean_absolute_error(y_test[:, 2], preds[:, 2])
	mae_total = (mae_gold + mae_silver + mae_bronze) / 3

	results = pd.DataFrame(
		{
			"team": [t for t, _ in test_meta],
			"year": [y for _, y in test_meta],
			"actual_gold": y_test[:, 0],
			"pred_gold": preds[:, 0],
			"actual_silver": y_test[:, 1],
			"pred_silver": preds[:, 1],
			"actual_bronze": y_test[:, 2],
			"pred_bronze": preds[:, 2],
		}
	)

	results = results.sort_values(["year", "team"])
	os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
	out_path = os.path.join(os.path.dirname(__file__), "outputs", "lstm_pred_vs_actual.csv")
	results.to_csv(out_path, index=False)

	print(f"Test years: {test_years}")
	print(f"Test MAE - Gold: {mae_gold:.3f}, Silver: {mae_silver:.3f}, Bronze: {mae_bronze:.3f}")
	print(f"Test MAE (average): {mae_total:.3f}")
	print(f"Saved predictions to {out_path}")
	print(results.head())


if __name__ == "__main__":
	main()
