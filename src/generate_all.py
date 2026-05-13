"""Generate all figures for the Semantic Visual-Inertial SLAM paper.

Usage:
    python generate_all.py              # generate all 10 figures
    python generate_all.py --fig 1      # generate only figure 1
    python generate_all.py --fig 1,3,5  # generate figures 1, 3, 5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig01_system_architecture import generate as gen_fig01
from fig02_kitti_trajectory import generate as gen_fig02
from fig03_euroc_trajectory import generate as gen_fig03
from fig04_yolov8_detection import generate as gen_fig04
from fig05_ablation_study import generate as gen_fig05
from fig06_timing_analysis import generate as gen_fig06
from fig07_sota_comparison import generate as gen_fig07
from fig08_qualitative_analysis import generate as gen_fig08
from fig09_failure_analysis import generate as gen_fig09
from fig10_parameter_sensitivity import generate as gen_fig10

GENERATORS = {
    1: ('System Architecture', gen_fig01),
    2: ('KITTI Trajectory', gen_fig02),
    3: ('EuRoC Trajectory', gen_fig03),
    4: ('YOLOv8 Detection', gen_fig04),
    5: ('Ablation Study', gen_fig05),
    6: ('Timing Analysis', gen_fig06),
    7: ('SOTA Comparison', gen_fig07),
    8: ('Qualitative Analysis', gen_fig08),
    9: ('Failure Analysis', gen_fig09),
    10: ('Parameter Sensitivity', gen_fig10),
}


def parse_args():
    selected = None
    if '--fig' in sys.argv:
        idx = sys.argv.index('--fig')
        if idx + 1 < len(sys.argv):
            selected = [int(x.strip()) for x in sys.argv[idx + 1].split(',')]
    return selected


def main():
    selected = parse_args()

    if selected:
        figs_to_gen = [(n, GENERATORS[n]) for n in selected if n in GENERATORS]
    else:
        figs_to_gen = list(GENERATORS.items())

    print(f'Generating {len(figs_to_gen)} figure(s)...')
    print('=' * 50)

    for num, (name, gen_func) in figs_to_gen:
        print(f'[{num}/10] Fig.{num}: {name} ...')
        try:
            gen_func()
        except Exception as e:
            print(f'  ERROR: {e}')

    print('=' * 50)
    print('Done.')


if __name__ == '__main__':
    main()
