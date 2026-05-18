import json
import os
from pathlib import Path


def is_image_file(name):
	_, ext = os.path.splitext(name)
	return ext.lower() in {".jpg", ".jpeg", ".png"}


def find_label_files(root_dir, label_name="labels.json"):
	root = Path(root_dir)
	if not root.exists():
		return []
	return list(root.rglob(label_name))


def summarize_label_file(labels_path, root_dir):
	with open(labels_path, "r", encoding="utf-8") as f:
		labels = json.load(f)

	images_meta = labels.get("images", {})
	label_files = set(images_meta.keys())

	images_dir = labels_path.parent
	disk_files = {
		p.name for p in images_dir.iterdir() if p.is_file() and is_image_file(p.name)
	}

	missing = disk_files - label_files
	extra = label_files - disk_files

	rel_dir = os.path.relpath(images_dir, root_dir)
	summary = {
		"rel_dir": rel_dir,
		"label_count": len(label_files),
		"file_count": len(disk_files),
		"missing": len(missing),
		"extra": len(extra),
	}
	status = "OK" if summary["label_count"] == summary["file_count"] else "DIFF"

	print(
		f"[{rel_dir}] labels={summary['label_count']} files={summary['file_count']} "
		f"missing={summary['missing']} extra={summary['extra']} => {status}"
	)

	return summary


def main():
	input_root = fr"data\processed\augmented\2026-05-15_11-18-07"
	label_name = "labels.json"

	label_files = find_label_files(input_root, label_name=label_name)
	if not label_files:
		print(f"No {label_name} found under {input_root}")
		return 1

	totals = {
		"label_count": 0,
		"file_count": 0,
		"missing": 0,
		"extra": 0,
	}

	for labels_path in label_files:
		summary = summarize_label_file(labels_path, input_root)
		totals["label_count"] += summary["label_count"]
		totals["file_count"] += summary["file_count"]
		totals["missing"] += summary["missing"]
		totals["extra"] += summary["extra"]

	status = "OK" if totals["label_count"] == totals["file_count"] else "DIFF"

	print(
		"\nTOTAL: "
		f"labels={totals['label_count']} files={totals['file_count']} "
		f"missing={totals['missing']} extra={totals['extra']} => {status}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
