#!/usr/bin/env python3
"""
TUM RGB-D Dataset Association Script
=====================================
Associates RGB and depth images by timestamp.
From: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools

Usage:
    python3 associate.py rgb.txt depth.txt > associate.txt
"""

import argparse


def parse_list(filename):
    """Read TUM format file, return list of (timestamp, path) tuples."""
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                data.append((float(parts[0]), parts[1]))
    return data


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """Match timestamps between two lists within max_difference."""
    matches = []
    best_matches = {}
    for a_ts, a_path in first_list:
        best_diff = max_difference
        best_b = None
        for b_ts, b_path in second_list:
            diff = abs(a_ts - (b_ts + offset))
            if diff < best_diff:
                best_diff = diff
                best_b = (b_ts, b_path)
        if best_b is not None:
            best_matches[a_ts] = best_b

    for a_ts, a_path in first_list:
        if a_ts in best_matches:
            b_ts, b_path = best_matches[a_ts]
            matches.append((a_ts, a_path, b_ts, b_path))

    matches.sort(key=lambda x: x[0])
    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Associate RGB and depth images')
    parser.add_argument('first', help='first text file (rgb.txt)')
    parser.add_argument('second', help='second text file (depth.txt)')
    parser.add_argument('--offset', type=float, default=0.0, help='time offset')
    parser.add_argument('--max_difference', type=float, default=0.02, help='max timestamp diff')
    args = parser.parse_args()

    first_list = parse_list(args.first)
    second_list = parse_list(args.second)
    matches = associate(first_list, second_list, args.offset, args.max_difference)

    for a_ts, a_path, b_ts, b_path in matches:
        print(f"{a_ts:.6f} {a_path} {b_ts:.6f} {b_path}")
