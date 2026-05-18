import json
import os
from pathlib import Path


def find_label_files(root_dir, label_name="labels.json"):
	root = Path(root_dir)
	if not root.exists():
		return []
	return list(root.rglob(label_name))


def extract_images_dict(labels):
	images = labels.get("images")
	if isinstance(images, dict):
		return images

	return {
		key: value
		for key, value in labels.items()
		if key not in {"version", "classes"} and isinstance(value, dict)
	}


def normalize_entry(meta):
	return {
		"year": meta.get("year"),
		"mint_mark": meta.get("mint_mark"),
	}


def main():
	input_root = fr"data\processed\augmented\2026-05-15_11-18-07"
	output_path = "data/annotations/labels.json"
	label_name = "labels.json"
	image_path_mode = "relative"

	label_files = find_label_files(input_root, label_name=label_name)
	if not label_files:
		print(f"No {label_name} found under {input_root}")
		return 1

	entries = []
	seen_paths = set()
	duplicates = 0

	for labels_path in label_files:
		with open(labels_path, "r", encoding="utf-8") as f:
			labels = json.load(f)

		images = extract_images_dict(labels)
		for filename, meta in images.items():
			if not isinstance(meta, dict):
				continue

			image_path = labels_path.parent / filename
			if image_path_mode == "relative":
				image_path_str = os.path.relpath(image_path, input_root)
				image_path_str = image_path_str.replace("\\", "/")
			else:
				image_path_str = str(image_path)

			if image_path_str in seen_paths:
				duplicates += 1
				continue
			seen_paths.add(image_path_str)

			entry = normalize_entry(meta)
			entry["image_name"] = filename
			entry["image_path"] = image_path_str
			entries.append(entry)

	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	output = sorted(entries, key=lambda item: item["image_path"])

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(output, f, indent=2)

	print(
		f"Wrote {output_path} | images={len(output)} | duplicates={duplicates}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
