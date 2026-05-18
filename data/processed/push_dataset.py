import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value


def load_labels(labels_path):
	with open(labels_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return data

	raise ValueError("labels.json must be a list of entries")


def load_split(split_path):
	with open(split_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return data

	raise ValueError(f"Split file must be a list: {split_path}")


def build_records(entries, image_root):
	root = Path(image_root)
	records = []
	missing = 0

	for entry in entries:
		image_name = entry.get("image_name")
		rel_path = entry.get("image_path")
		if not image_name or not rel_path:
			continue

		full_path = root / rel_path
		if not full_path.is_file():
			missing += 1
			continue

		records.append(
			{
				"image": str(full_path),
				"image_name": image_name,
				"year": entry.get("year"),
				"mint_mark": entry.get("mint_mark"),
			}
		)

	return records, missing


def main():
	splits_dir = "data/splits"
	image_root = "data/processed/augmented/2026-05-15_11-18-07"
	repo_id = "BrownAtLutech/coin_mintMark_year_detection"
	private = False
	token = os.getenv("HF_TOKEN")
	
	split_files = {
		"train": os.path.join(splits_dir, "train.json"),
		"validation": os.path.join(splits_dir, "validation.json"),
		"test": os.path.join(splits_dir, "test.json"),
	}

	if not all(os.path.isfile(path) for path in split_files.values()):
		print("Missing split files. Expecting train.json, validation.json, test.json")
		return 1

	features = Features(
		{
			"image": Image(),
			"image_name": Value("string"),
			"year": Value("string"),
			"mint_mark": Value("string"),
		}
	)

	datasets = {}
	missing_total = 0
	for split_name, split_path in split_files.items():
		split_entries = load_split(split_path)
		records, missing = build_records(split_entries, image_root)
		missing_total += missing
		if not records:
			print(f"No valid records found for split: {split_name}")
			return 1
		datasets[split_name] = Dataset.from_list(records, features=features)

	dataset_dict = DatasetDict(datasets)
	dataset_dict.push_to_hub(repo_id, private=private, token=token)

	print(
		f"Pushed splits to {repo_id} | "
		f"train={len(datasets['train'])} val={len(datasets['validation'])} "
		f"test={len(datasets['test'])} missing_images={missing_total}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
