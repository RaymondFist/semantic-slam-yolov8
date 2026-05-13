"""Figure 8: Qualitative Analysis - Real Trajectory Data"""
import matplotlib.pyplot as plt
import numpy as np
import os
from common import output_dir, data_dir, save_real_data, COLORS


def load_tum_trajectory(filepath):
    stamps, xyz = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                stamps.append(float(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(stamps), np.array(xyz)


def load_kitti_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) >= 12:
                tx, ty, tz = values[3], values[7], values[11]
                poses.append([tx, ty, tz])
    return np.array(poses)


def generate():
    euroc_dir = os.path.join(data_dir, 'real_trajectories', 'EuRoC')
    kitti_dir = os.path.join(data_dir, 'real_trajectories', 'KITTI', 'dataset', 'poses')

    stamps_mh01, gt_mh01 = load_tum_trajectory(os.path.join(euroc_dir, 'MH_01_easy.txt'))
    stamps_mh05, gt_mh05 = load_tum_trajectory(os.path.join(euroc_dir, 'MH_05_difficult.txt'))
    gt_kitti_00 = load_kitti_poses(os.path.join(kitti_dir, '00.txt'))
    gt_kitti_05 = load_kitti_poses(os.path.join(kitti_dir, '05.txt'))

    t_mh01 = stamps_mh01 - stamps_mh01[0]
    t_mh05 = stamps_mh05 - stamps_mh05[0]

    save_real_data('fig08_qualitative_analysis', {
        't_mh01': t_mh01.tolist(), 'gt_mh01_x': gt_mh01[:, 0].tolist(),
        'gt_mh01_y': gt_mh01[:, 1].tolist(), 'gt_mh01_z': gt_mh01[:, 2].tolist(),
        't_mh05': t_mh05.tolist(), 'gt_mh05_x': gt_mh05[:, 0].tolist(),
        'gt_mh05_y': gt_mh05[:, 1].tolist(), 'gt_mh05_z': gt_mh05[:, 2].tolist(),
        'gt_kitti_00_x': gt_kitti_00[:, 0].tolist(), 'gt_kitti_00_y': gt_kitti_00[:, 1].tolist(),
        'gt_kitti_05_x': gt_kitti_05[:, 0].tolist(), 'gt_kitti_05_y': gt_kitti_05[:, 1].tolist(),
    }, {
        'ground_truth': 'KITTI (Geiger et al., 2012) and EuRoC MAV (Burri et al., 2016) datasets.',
        'note': 'Qualitative trajectory visualization using real ground truth data.',
    })

    fig = plt.figure(figsize=(14, 9))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(gt_kitti_00[:, 0], gt_kitti_00[:, 1], 'b-', lw=0.8, label='KITTI 00')
    ax1.plot(gt_kitti_05[:, 0], gt_kitti_05[:, 1], 'r-', lw=0.8, label='KITTI 05')
    ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
    ax1.set_title('(a) KITTI Ground Truth Trajectories', fontsize=10)
    ax1.legend(fontsize=7); ax1.set_aspect('equal')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(gt_mh01[:, 0], gt_mh01[:, 1], 'b-', lw=0.8, label='MH_01_easy')
    ax2.plot(gt_mh05[:, 0], gt_mh05[:, 1], 'r-', lw=0.8, label='MH_05_difficult')
    ax2.set_xlabel('X [m]'); ax2.set_ylabel('Y [m]')
    ax2.set_title('(b) EuRoC Ground Truth Trajectories', fontsize=10)
    ax2.legend(fontsize=7); ax2.set_aspect('equal')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t_mh01, gt_mh01[:, 0], 'b-', lw=1, label='MH_01 X')
    ax3.plot(t_mh01, gt_mh01[:, 1], 'b--', lw=1, label='MH_01 Y')
    ax3.plot(t_mh01, gt_mh01[:, 2], 'b:', lw=1, label='MH_01 Z')
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Position [m]')
    ax3.set_title('(c) MH_01 Position vs Time', fontsize=10)
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t_mh05, gt_mh05[:, 0], 'r-', lw=1, label='MH_05 X')
    ax4.plot(t_mh05, gt_mh05[:, 1], 'r--', lw=1, label='MH_05 Y')
    ax4.plot(t_mh05, gt_mh05[:, 2], 'r:', lw=1, label='MH_05 Z')
    ax4.set_xlabel('Time [s]'); ax4.set_ylabel('Position [m]')
    ax4.set_title('(d) MH_05 Position vs Time', fontsize=10)
    ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(2, 3, 5)
    kitti_00_dist = np.sqrt(np.diff(gt_kitti_00[:, 0])**2 + np.diff(gt_kitti_00[:, 1])**2)
    kitti_05_dist = np.sqrt(np.diff(gt_kitti_05[:, 0])**2 + np.diff(gt_kitti_05[:, 1])**2)
    ax5.hist(kitti_00_dist, bins=50, alpha=0.5, label='KITTI 00', color='blue')
    ax5.hist(kitti_05_dist, bins=50, alpha=0.5, label='KITTI 05', color='red')
    ax5.set_xlabel('Frame-to-Frame Distance [m]'); ax5.set_ylabel('Frequency')
    ax5.set_title('(e) Motion Distribution', fontsize=10)
    ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary = (
        'Qualitative Analysis\n'
        '='*22 + '\n'
        'KITTI 00: Urban driving\n'
        f'  Length: {len(gt_kitti_00)} frames\n'
        f'  Range X: [{gt_kitti_00[:,0].min():.0f}, {gt_kitti_00[:,0].max():.0f}] m\n'
        f'  Range Y: [{gt_kitti_00[:,1].min():.0f}, {gt_kitti_00[:,1].max():.0f}] m\n\n'
        'KITTI 05: Urban driving\n'
        f'  Length: {len(gt_kitti_05)} frames\n'
        f'  Range X: [{gt_kitti_05[:,0].min():.0f}, {gt_kitti_05[:,0].max():.0f}] m\n'
        f'  Range Y: [{gt_kitti_05[:,1].min():.0f}, {gt_kitti_05[:,1].max():.0f}] m\n\n'
        'EuRoC MH_01: Indoor flight\n'
        f'  Duration: {t_mh01[-1]:.1f} s\n'
        f'  Altitude range: [{gt_mh01[:,2].min():.1f}, {gt_mh01[:,2].max():.1f}] m\n\n'
        'EuRoC MH_05: Indoor flight\n'
        f'  Duration: {t_mh05[-1]:.1f} s\n'
        f'  Altitude range: [{gt_mh05[:,2].min():.1f}, {gt_mh05[:,2].max():.1f}] m'
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=6.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Qualitative Trajectory Analysis (Real Ground Truth Data)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig08_qualitative_analysis.png'))
    plt.close()
    print('Fig.8: Qualitative Analysis - Done')


if __name__ == '__main__':
    generate()
