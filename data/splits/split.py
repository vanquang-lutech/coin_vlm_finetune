import json
import os
import random
from collections import defaultdict


def load_labels(path):
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return data

	if isinstance(data, dict) and "images" in data:
		return list(data["images"].values())

	if isinstance(data, dict):
		entries = []
		for key, value in data.items():
			if key in {"version", "classes"}:
				continue
			if isinstance(value, dict):
				entry = dict(value)
				entry.setdefault("image_name", key)
				entries.append(entry)
		return entries

	return []


def split_counts(n, ratios):
	raw = [n * r for r in ratios]
	counts = [int(x) for x in raw]
	remainder = n - sum(counts)

	if remainder > 0:
		fractional = [x - int(x) for x in raw]
		order = sorted(range(len(fractional)), key=lambda i: fractional[i], reverse=True)
		for i in range(remainder):
			counts[order[i % len(counts)]] += 1

	return counts


def stratified_split(entries, key_fn, ratios, seed=42):
	rng = random.Random(seed)
	buckets = defaultdict(list)
	for entry in entries:
		buckets[key_fn(entry)].append(entry)

	train = []
	val = []
	test = []

	for group_entries in buckets.values():
		rng.shuffle(group_entries)
		n = len(group_entries)
		train_count, val_count, test_count = split_counts(n, ratios)

		train.extend(group_entries[:train_count])
		val.extend(group_entries[train_count:train_count + val_count])
		test.extend(group_entries[train_count + val_count:train_count + val_count + test_count])

	rng.shuffle(train)
	rng.shuffle(val)
	rng.shuffle(test)
	return train, val, test


def main():
	input_path = "data/annotations/labels.json"
	output_dir = "data/splits"
	ratios = (0.8, 0.1, 0.1)
	seed = 42

	entries = load_labels(input_path)
	if not entries:
		print(f"No entries found in {input_path}")
		return 1

	def key_fn(entry):
		return entry.get("mint_mark") or "__NONE__"

	train, val, test = stratified_split(entries, key_fn, ratios, seed=seed)

	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
		json.dump(train, f, indent=2)
	with open(os.path.join(output_dir, "validation.json"), "w", encoding="utf-8") as f:
		json.dump(val, f, indent=2)
	with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
		json.dump(test, f, indent=2)

	print(
		f"Done. train={len(train)} val={len(val)} test={len(test)} "
		f"(total={len(entries)})"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
