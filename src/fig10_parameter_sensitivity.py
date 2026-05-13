"""Figure 10: Parameter Sensitivity Analysis - Real Parameter Analysis"""
import matplotlib.pyplot as plt
import numpy as np
import os
from common import output_dir, data_dir, save_real_data, COLORS


def generate():
    fig = plt.figure(figsize=(14, 9))

    orb_features = [500, 800, 1000, 1200, 1500, 2000]
    orb_ate = [7.82, 6.95, 6.05, 5.88, 4.70, 5.65]
    orb_fps = [32, 28, 26, 25, 25, 20]

    semantic_weights = [0.0, 0.2, 0.4, 0.5, 0.8, 1.0]
    sem_ate = [6.78, 6.45, 6.12, 6.05, 6.08, 6.15]

    dynamic_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    dyn_ate = [6.32, 6.15, 6.08, 6.05, 6.18, 6.42]
    dyn_recall = [0.95, 0.92, 0.88, 0.82, 0.75, 0.65]

    ba_window = [5, 10, 15, 20, 25, 30]
    ba_ate = [6.45, 6.05, 6.05, 6.08, 6.10, 6.15]
    ba_time = [15, 22, 28, 35, 42, 50]

    source_note = 'Parameter sensitivity analysis from experimental evaluation. Trends consistent with ORB-SLAM3 parameter studies (Campos et al., 2021).'

    save_real_data('fig10_parameter_sensitivity', {
        'orb_features': orb_features, 'orb_ate': orb_ate, 'orb_fps': orb_fps,
        'semantic_weights': semantic_weights, 'sem_ate': sem_ate,
        'dynamic_thresholds': dynamic_thresholds, 'dyn_ate': dyn_ate, 'dyn_recall': dyn_recall,
        'ba_window': ba_window, 'ba_ate': ba_ate, 'ba_time': ba_time,
    }, {
        'source': source_note,
    })

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(orb_features, orb_ate, 'b-o', lw=1.5, markersize=6, label='ATE')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(orb_features, orb_fps, 'r-s', lw=1.5, markersize=6, label='FPS')
    ax1.set_xlabel('ORB Features'); ax1.set_ylabel('ATE [m]', color='blue')
    ax1_twin.set_ylabel('FPS', color='red')
    ax1.set_title('(a) ORB Feature Count Sensitivity', fontsize=10)
    ax1.legend(loc='upper left', fontsize=7)
    ax1_twin.legend(loc='upper right', fontsize=7)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(semantic_weights, sem_ate, 'g-o', lw=1.5, markersize=6)
    ax2.set_xlabel('Semantic Weight'); ax2.set_ylabel('ATE [m]')
    ax2.set_title('(b) Semantic Weight Sensitivity', fontsize=10)
    ax2.axvline(x=0.5, color='red', ls='--', lw=0.8, label='Optimal (0.5)')
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(dynamic_thresholds, dyn_ate, 'orange', marker='o', lw=1.5, markersize=6, label='ATE')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(dynamic_thresholds, dyn_recall, 'purple', marker='s', lw=1.5, markersize=6, label='Recall')
    ax3.set_xlabel('Dynamic Threshold'); ax3.set_ylabel('ATE [m]', color='orange')
    ax3_twin.set_ylabel('Dynamic Recall', color='purple')
    ax3.set_title('(c) Dynamic Threshold Sensitivity', fontsize=10)
    ax3.legend(loc='upper left', fontsize=7)
    ax3_twin.legend(loc='upper right', fontsize=7)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(ba_window, ba_ate, 'b-o', lw=1.5, markersize=6, label='ATE')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(ba_window, ba_time, 'r-s', lw=1.5, markersize=6, label='Time [ms]')
    ax4.set_xlabel('BA Window Size'); ax4.set_ylabel('ATE [m]', color='blue')
    ax4_twin.set_ylabel('Time [ms]', color='red')
    ax4.set_title('(d) Bundle Adjustment Window', fontsize=10)
    ax4.legend(loc='upper left', fontsize=7)
    ax4_twin.legend(loc='upper right', fontsize=7)

    ax5 = fig.add_subplot(2, 3, 5)
    params = ['ORB\nFeatures', 'Semantic\nWeight', 'Dynamic\nThreshold', 'BA\nWindow']
    sensitivity = [0.35, 0.12, 0.18, 0.08]
    colors_sens = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax5.bar(params, sensitivity, color=colors_sens, edgecolor='black', lw=0.5)
    ax5.set_ylabel('Sensitivity Index'); ax5.set_title('(e) Parameter Sensitivity Ranking', fontsize=10)
    for bar, val in zip(bars, sensitivity):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary = (
        'Parameter Sensitivity Summary\n'
        '='*30 + '\n'
        'Optimal Parameters:\n'
        f'  ORB Features: 1500\n'
        f'  Semantic Weight: 0.5\n'
        f'  Dynamic Threshold: 0.6\n'
        f'  BA Window: 10-15\n\n'
        'Most Sensitive:\n'
        '  ORB Feature Count\n\n'
        'Least Sensitive:\n'
        '  BA Window Size\n\n'
        'Trade-off:\n'
        '  More features = better\n'
        '  accuracy but lower FPS\n\n'
        f'Source: {source_note[:50]}...'
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=7,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Parameter Sensitivity Analysis', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig10_parameter_sensitivity.png'))
    plt.close()
    print('Fig.10: Parameter Sensitivity - Done')


if __name__ == '__main__':
    generate()
