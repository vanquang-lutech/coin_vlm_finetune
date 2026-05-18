import os
import json
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm
import datetime

random.seed(42)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def angular_distance(a, b):
    """Calculate the smallest angular distance between two angles in degrees."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def random_rotate_image(image, angle_range=(-180, 180), num_augmented=2, min_angle_gap=30):
    """
    Rotate the input image by randomly selected angles within range.
    Args:
        image: Input image.
        angle_range: range of angles (min, max) in degrees.
        num_augmented: number of augmented images to generate.
        min_angle_gap: min angular distance in degrees between selected angles and 0.
    Returns:
        List of (angle, rotated_image, M) tuples.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    candidates = [
        a for a in range(*angle_range)
        if angular_distance(a, 0) >= min_angle_gap
    ]
    random.shuffle(candidates)

    selected_angles = []
    for candidate in candidates:
        if len(selected_angles) >= num_augmented:
            break
        if all(angular_distance(candidate, a) >= min_angle_gap for a in selected_angles):
            selected_angles.append(candidate)

    if len(selected_angles) < num_augmented:
        raise ValueError(
            f"Cannot create {num_augmented} angles with min_gap={min_angle_gap} "
            f"in range {angle_range}. Please reduce min_angle_gap or num_augmented."
        )

    results = []
    for angle in selected_angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        results.append((angle, rotated, M))

    return results


def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in {".jpg", ".jpeg", ".png"}


def rotate_bbox(bbox, M, width, height):
    """
    Rotate a bounding box using the same rotation matrix M used for the image.
    Args:
        bbox: [x1, y1, x2, y2] or None.
        M: 2x3 rotation matrix.
        width: image width.
        height: image height.
    Returns:
        Rotated [x1, y1, x2, y2] or None if invalid.
    """
    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )

    ones = np.ones((4, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])
    rotated = (M @ corners_h.T).T

    new_x1 = int(round(rotated[:, 0].min()))
    new_y1 = int(round(rotated[:, 1].min()))
    new_x2 = int(round(rotated[:, 0].max()))
    new_y2 = int(round(rotated[:, 1].max()))

    new_x1 = max(0, min(new_x1, width - 1))
    new_y1 = max(0, min(new_y1, height - 1))
    new_x2 = max(0, min(new_x2, width - 1))
    new_y2 = max(0, min(new_y2, height - 1))

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None

    return [new_x1, new_y1, new_x2, new_y2]


def is_valid_bbox(bbox, orig_bbox, max_area_ratio=2.0):
    """Check if rotated bbox is not too large or degenerate compared to original."""
    if bbox is None or orig_bbox is None:
        return False
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    orig_area = (orig_bbox[2] - orig_bbox[0]) * (orig_bbox[3] - orig_bbox[1])
    if orig_area == 0:
        return False
    return 0 < area <= orig_area * max_area_ratio


def build_augmented_name(filename, angle):
    base, ext = os.path.splitext(filename)
    sign = "p" if angle >= 0 else "m"
    return f"{base}_rot{sign}{abs(int(angle))}{ext}"


def find_image_dirs(input_root):
    for root, _, files in os.walk(input_root):
        if "labels.json" in files:
            yield root


def main():
    input_root = "data/raw"
    output_root = "data/processed/augmented/" + timestamp
    angle_range = (-180, 180)
    num_augmented = 2
    min_angle_gap = 30
    copy_originals = True

    image_dirs = list(find_image_dirs(input_root))
    if not image_dirs:
        print(f"No labels.json found under {input_root}")
        return

    total_created = 0

    for images_dir in tqdm(image_dirs, desc="Folders"):
        labels_path = os.path.join(images_dir, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        rel_dir = os.path.relpath(images_dir, input_root)
        output_dir = os.path.join(output_root, rel_dir)
        os.makedirs(output_dir, exist_ok=True)

        updated_images = {}
        images = labels.get("images", {})

        for filename, meta in tqdm(images.items(), desc=rel_dir, leave=False):
            if not is_image_file(filename):
                continue

            src_path = os.path.join(images_dir, filename)
            if not os.path.isfile(src_path):
                continue

            image = cv2.imread(src_path)
            if image is None:
                continue

            if copy_originals:
                dst_path = os.path.join(output_dir, filename)
                if not os.path.isfile(dst_path):
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                updated_images[filename] = dict(meta)

            height, width = image.shape[:2]

            for angle, rotated, M in random_rotate_image(
                image,
                angle_range=angle_range,
                num_augmented=num_augmented,
                min_angle_gap=min_angle_gap,
            ):
                new_name = build_augmented_name(filename, angle)
                dst_path = os.path.join(output_dir, new_name)
                if os.path.isfile(dst_path):
                    continue
                if not cv2.imwrite(dst_path, rotated):
                    continue

                new_meta = dict(meta)

                if "yearCoord" in new_meta:
                    orig_bbox = new_meta["yearCoord"]
                    rotated_bbox = rotate_bbox(orig_bbox, M, width, height)
                    new_meta["yearCoord"] = (
                        rotated_bbox if is_valid_bbox(rotated_bbox, orig_bbox) else None
                    )

                if "mintMarkCoord" in new_meta:
                    orig_bbox = new_meta["mintMarkCoord"]
                    rotated_bbox = rotate_bbox(orig_bbox, M, width, height)
                    new_meta["mintMarkCoord"] = (
                        rotated_bbox if is_valid_bbox(rotated_bbox, orig_bbox) else None
                    )

                updated_images[new_name] = new_meta
                total_created += 1

        updated_labels = dict(labels)
        updated_labels["images"] = updated_images
        with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(updated_labels, f, indent=2)

        print(f"  {rel_dir}: {len(updated_images)} images (originals + augmented)")

    print(f"\nDone! Total augmented images created: {total_created}")


if __name__ == "__main__":
    main()