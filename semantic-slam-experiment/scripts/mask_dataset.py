#!/usr/bin/env python3
"""
数据集动态区域掩码脚本 (Plan C: Image Masking)
==============================================
读取 YOLOv8 检测 JSON，将动态物体区域在 RGB 图像中置零（黑色），
生成掩码后的数据集副本。掩码后的数据集直接用原生 ORB-SLAM3 运行，
无需任何 SemanticSLAM patch。

支持: TUM RGB-D, KITTI, EuRoC

用法:
    python3 scripts/mask_dataset.py \
        --detections data/detections/walking_xyz \
        --dataset   data/datasets/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --output    data/datasets_masked/TUM/rgbd_dataset_freiburg3_walking_xyz \
        --dataset-type tum

    # 批量处理:
    python3 scripts/mask_dataset.py --all --datasets data/datasets --detections data/detections --output data/datasets_masked
"""

import os
import sys
import json
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path


# 动态 COCO 类别 ID
DYNAMIC_COCO_IDS = {0, 1, 2, 3, 5, 7, 16, 17}


def load_detection_map(detection_dir):
    """加载所有检测 JSON，建立 frame_id -> detections 的映射。"""
    det_map = {}
    if not os.path.isdir(detection_dir):
        print(f"  [WARN] 检测目录不存在: {detection_dir}")
        return det_map

    for fname in os.listdir(detection_dir):
        if not fname.endswith('.json'):
            continue
        try:
            frame_id = int(fname.replace('.json', ''))
        except ValueError:
            continue

        fpath = os.path.join(detection_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            det_map[frame_id] = data.get("detections", [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] 读取失败 {fpath}: {e}")

    return det_map


def mask_image(image, detections, dilate_px=3):
    """将动态物体 bbox 区域在图像中置零。

    Args:
        image: BGR numpy array
        detections: YOLO detection list
        dilate_px: bbox 向外扩展像素数（避免边缘特征残留）

    Returns:
        masked image (in-place modified)
    """
    h, w = image.shape[:2]

    for det in detections:
        if not det.get("is_dynamic", False):
            continue

        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue

        x1 = max(0, int(bbox[0]) - dilate_px)
        y1 = max(0, int(bbox[1]) - dilate_px)
        x2 = min(w, int(bbox[2]) + dilate_px)
        y2 = min(h, int(bbox[3]) + dilate_px)

        if x2 > x1 and y2 > y1:
            image[y1:y2, x1:x2] = 0

    return image


def mask_dataset_tum(dataset_dir, detection_dir, output_dir):
    """掩码 TUM RGB-D 数据集。

    TUM 目录结构:
        rgb/              ← RGB 图像（需要掩码）
        depth/            ← 深度图（直接复制/链接）
        rgb.txt           ← 帧列表
        depth.txt
        associate.txt
        groundtruth.txt
        accelerometer.txt (可选)
    """
    os.makedirs(output_dir, exist_ok=True)

    det_map = load_detection_map(detection_dir)
    if not det_map:
        print(f"  [ERROR] 无检测数据，跳过")
        return False

    rgb_dir = os.path.join(dataset_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"  [ERROR] RGB 目录不存在: {rgb_dir}")
        return False

    # 创建输出 RGB 目录
    output_rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(output_rgb_dir, exist_ok=True)

    # 复制/链接其他文件
    for item in ["depth", "rgb.txt", "depth.txt", "associate.txt",
                 "groundtruth.txt", "accelerometer.txt"]:
        src = os.path.join(dataset_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    # 深度图目录 — 不需要修改，直接复制符号链接或用硬链接
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copytree(src, dst)
            else:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # 读取 rgb.txt 获取帧顺序
    rgb_txt = os.path.join(dataset_dir, "rgb.txt")
    frames = []
    if os.path.isfile(rgb_txt):
        with open(rgb_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    frames.append((float(parts[0]), parts[1]))

    print(f"  处理 {len(frames)} 帧...")

    masked_count = 0
    for idx, (timestamp, rgb_path) in enumerate(frames):
        src_path = os.path.join(dataset_dir, rgb_path)
        if not os.path.isfile(src_path):
            continue

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # frame_id 对应 detection JSON 的文件名编号
        dets = det_map.get(idx, [])

        if dets:
            mask_image(img, dets)
            masked_count += 1

        # 保存掩码后的图像
        dst_path = os.path.join(output_dir, rgb_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img)

        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(frames)} 帧完成 ({masked_count} 帧含动态物体)")

    print(f"  完成: {len(frames)} 帧, {masked_count} 帧含动态物体被掩码")
    return True


def mask_dataset_kitti(dataset_dir, detection_dir, output_dir):
    """掩码 KITTI Odometry 数据集。

    KITTI 目录结构:
        image_0/    ← 左目灰度图 (需要掩码)
        image_1/    ← 右目灰度图 (需要掩码)
        times.txt
        calib.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    det_map = load_detection_map(detection_dir)
    if not det_map:
        print(f"  [ERROR] 无检测数据，跳过")
        return False

    for img_dir_name in ["image_0", "image_1"]:
        src_dir = os.path.join(dataset_dir, img_dir_name)
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(output_dir, img_dir_name)
        os.makedirs(dst_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
        print(f"  处理 {img_dir_name}: {len(images)} 帧...")

        masked_count = 0
        for img_name in images:
            frame_id = int(img_name.replace('.png', ''))
            src_path = os.path.join(src_dir, img_name)

            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            dets = det_map.get(frame_id, [])
            if dets:
                # 灰度图需要转 BGR 才能用颜色赋值, 但黑色在灰度图也是 0
                mask_image(img, dets)
                masked_count += 1

            dst_path = os.path.join(dst_dir, img_name)
            cv2.imwrite(dst_path, img)

        print(f"    {img_dir_name} 完成: {masked_count} 帧含动态物体被掩码")

    # 复制其他文件
    for item in ["times.txt", "calib.txt"]:
        src = os.path.join(dataset_dir, item)
        if os.path.exists(src):
            dst = os.path.join(output_dir, item)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    return True


def mask_dataset_euroc(dataset_dir, detection_dir, output_dir):
    """掩码 EuRoC MAV 数据集。

    EuRoC 目录结构:
        mav0/cam0/data/   ← 灰度图 (需要掩码)
        mav0/cam0/sensor.yaml
        mav0/cam1/data/   ← 灰度图 (可选)
        mav0/imu0/data.csv
        ...
    """
    os.makedirs(output_dir, exist_ok=True)

    det_map = load_detection_map(detection_dir)
    if not det_map:
        print(f"  [ERROR] 无检测数据，跳过")
        return False

    for cam_name in ["cam0", "cam1"]:
        src_dir = os.path.join(dataset_dir, "mav0", cam_name, "data")
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(output_dir, "mav0", cam_name, "data")
        os.makedirs(dst_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
        print(f"  处理 mav0/{cam_name}/data: {len(images)} 帧...")

        masked_count = 0
        for img_name in images:
            frame_id = int(img_name.replace('.png', ''))
            src_path = os.path.join(src_dir, img_name)

            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            dets = det_map.get(frame_id, [])
            if dets:
                mask_image(img, dets)
                masked_count += 1

            dst_path = os.path.join(dst_dir, img_name)
            cv2.imwrite(dst_path, img)

        print(f"    mav0/{cam_name}/data 完成: {masked_count} 帧含动态物体被掩码")

    # 复制 sensor.yaml
    for cam_name in ["cam0", "cam1"]:
        src = os.path.join(dataset_dir, "mav0", cam_name, "sensor.yaml")
        if os.path.isfile(src):
            dst = os.path.join(output_dir, "mav0", cam_name, "sensor.yaml")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # 复制 IMU 和地面真值
    for sub in ["imu0", "state_groundtruth_estimate0"]:
        src = os.path.join(dataset_dir, "mav0", sub)
        if os.path.isdir(src):
            dst = os.path.join(output_dir, "mav0", sub)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copytree(src, dst)

    return True


def detect_dataset_type(dataset_dir):
    """自动检测数据集类型。"""
    if os.path.isdir(os.path.join(dataset_dir, "mav0", "cam0", "data")):
        return "euroc"
    elif os.path.isdir(os.path.join(dataset_dir, "image_0")):
        return "kitti"
    elif os.path.isfile(os.path.join(dataset_dir, "rgb.txt")):
        return "tum"
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="数据集动态区域掩码 (Plan C)")
    parser.add_argument("--detections", help="检测 JSON 目录")
    parser.add_argument("--dataset", help="原始数据集目录")
    parser.add_argument("--output", help="掩码后数据集输出目录")
    parser.add_argument("--dataset-type", choices=["tum", "kitti", "euroc", "auto"],
                        default="auto", help="数据集类型 (默认自动检测)")
    parser.add_argument("--all", action="store_true",
                        help="批量处理所有已知序列")
    parser.add_argument("--datasets", default="data/datasets",
                        help="数据集根目录 (--all 模式)")
    parser.add_argument("--detections-root", default="data/detections",
                        help="检测结果根目录 (--all 模式)")
    args = parser.parse_args()

    if args.all:
        # 批量处理模式
        return batch_mask_all(args)

    # 单序列处理模式
    if not args.detections or not args.dataset or not args.output:
        parser.error("单序列模式需要 --detections, --dataset, --output")

    ds_type = args.dataset_type
    if ds_type == "auto":
        ds_type = detect_dataset_type(args.dataset)

    if ds_type is None:
        print(f"[ERROR] 无法检测数据集类型: {args.dataset}")
        sys.exit(1)

    print(f"掩码数据集: {args.dataset} → {args.output}")
    print(f"  类型: {ds_type}")
    print(f"  检测: {args.detections}")

    if ds_type == "tum":
        ok = mask_dataset_tum(args.dataset, args.detections, args.output)
    elif ds_type == "kitti":
        ok = mask_dataset_kitti(args.dataset, args.detections, args.output)
    elif ds_type == "euroc":
        ok = mask_dataset_euroc(args.dataset, args.detections, args.output)
    else:
        print(f"[ERROR] 不支持的数据集类型: {ds_type}")
        sys.exit(1)

    if ok:
        print(f"\n掩码完成: {args.output}")
    else:
        print(f"\n掩码失败")
        sys.exit(1)


def batch_mask_all(args):
    """批量处理所有已知的训练/评估序列。"""
    datasets_root = args.datasets
    detections_root = args.detections_root
    output_root = os.path.join(os.path.dirname(datasets_root), "datasets_masked")

    # 定义所有序列: (数据集类型, 数据集路径, 检测目录名)
    sequences = [
        # TUM
        ("tum", "TUM/rgbd_dataset_freiburg3_sitting_static", "sitting_static"),
        ("tum", "TUM/rgbd_dataset_freiburg3_sitting_xyz", "sitting_xyz"),
        ("tum", "TUM/rgbd_dataset_freiburg3_walking_static", "walking_static"),
        ("tum", "TUM/rgbd_dataset_freiburg3_walking_xyz", "walking_xyz"),
        ("tum", "TUM/rgbd_dataset_freiburg3_walking_halfsphere", "walking_halfsphere"),
        # KITTI
        ("kitti", "KITTI/00", "kitti_00"),
        # EuRoC
        ("euroc", "EuRoC/MH_01_easy", "euroc_mh01easy"),
        ("euroc", "EuRoC/MH_03_medium", "euroc_mh03medium"),
        ("euroc", "EuRoC/MH_05_difficult", "euroc_mh05difficult"),
    ]

    total = 0
    success = 0
    for ds_type, ds_rel_path, det_name in sequences:
        ds_path = os.path.join(datasets_root, ds_rel_path)
        det_path = os.path.join(detections_root, det_name)
        out_path = os.path.join(output_root, ds_rel_path)

        if not os.path.isdir(ds_path):
            print(f"\n[SKIP] 数据集不存在: {ds_path}")
            continue
        if not os.path.isdir(det_path):
            print(f"\n[SKIP] 检测数据不存在: {det_path}")
            continue

        total += 1
        print(f"\n{'='*60}")
        print(f"[{total}] {ds_type.upper()}: {ds_rel_path}")
        print(f"{'='*60}")

        if ds_type == "tum":
            ok = mask_dataset_tum(ds_path, det_path, out_path)
        elif ds_type == "kitti":
            ok = mask_dataset_kitti(ds_path, det_path, out_path)
        elif ds_type == "euroc":
            ok = mask_dataset_euroc(ds_path, det_path, out_path)
        else:
            ok = False

        if ok:
            success += 1

    print(f"\n{'='*60}")
    print(f"批量掩码完成: {success}/{total} 成功")
    print(f"输出目录: {output_root}")
    print(f"{'='*60}")

    return 0 if success == total else 1


if __name__ == "__main__":
    main()