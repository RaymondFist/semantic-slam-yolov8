"""Figure 1: System Architecture"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from common import output_dir


def draw_box(ax, x, y, w, h, text, color, fontsize=8):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          facecolor=color, edgecolor='#37474F', linewidth=1.5,
                          zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='#263238', zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color='#546E7A', lw=1.8, style='simple',
               zorder=1, connectionstyle=None):
    arrowstyle_map = {
        'simple': '->',
        'head': '-|>',
    }
    arrow_kw = dict(arrowstyle=arrowstyle_map.get(style, '->'),
                    color=color, lw=lw, shrinkA=0, shrinkB=0)
    if connectionstyle:
        arrow_kw['connectionstyle'] = connectionstyle

    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=arrow_kw, zorder=zorder)


def generate():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # ============================================================
    # Box definitions: (x, y, w, h, text, color)
    # All coordinates carefully chosen so arrows connect edge-to-edge
    # ============================================================

    # --- Top row: Main SLAM Pipeline (y ~ 4.8 to 6.2) ---
    B1 = (0.5, 4.8, 2.6, 1.4, 'Stereo/RGB-D Camera\n+ IMU Data', '#E3F2FD')
    B2 = (4.0, 4.8, 2.6, 1.4, 'ORB Feature\nExtraction', '#BBDEFB')
    B3 = (7.5, 4.8, 2.6, 1.4, 'Visual-Inertial\nOdometry', '#90CAF9')
    B4 = (11.0, 4.8, 2.5, 1.4, 'Pose\nEstimation', '#64B5F6')

    # --- Middle row: Dynamic Object Pipeline (y ~ 2.1 to 3.5) ---
    B5 = (4.0, 2.1, 2.6, 1.4, 'YOLOv8-nano\nObject Detection', '#FFCCBC')
    B6 = (7.5, 2.1, 2.6, 1.4, 'Optical Flow\nVerification', '#FFAB91')
    B7 = (11.0, 2.1, 2.5, 1.4, 'Dynamic Feature\nFiltering', '#FF8A65')

    # --- Bottom row: Optimization & Map (y ~ 0.3 to 1.5) ---
    B8 = (7.5, 0.3, 2.6, 1.2, 'Semantic Weight\nOptimization', '#C8E6C9')
    B9 = (0.5, 0.3, 2.6, 1.2, 'Loop Closure &\nMap Management', '#E1BEE7')

    boxes = [B1, B2, B3, B4, B5, B6, B7, B8, B9]

    for x, y, w, h, text, color in boxes:
        draw_box(ax, x, y, w, h, text, color)

    # ============================================================
    # Edge coordinates for each box
    # ============================================================
    def edges(x, y, w, h):
        return {
            'left': x,
            'right': x + w,
            'top': y + h,
            'bottom': y,
            'cx': x + w / 2,
            'cy': y + h / 2,
        }

    e1 = edges(*B1[:4])
    e2 = edges(*B2[:4])
    e3 = edges(*B3[:4])
    e4 = edges(*B4[:4])
    e5 = edges(*B5[:4])
    e6 = edges(*B6[:4])
    e7 = edges(*B7[:4])
    e8 = edges(*B8[:4])
    e9 = edges(*B9[:4])

    # ============================================================
    # Arrows: ALL from box edge to box edge (never inside a box)
    # ============================================================

    # --- Main SLAM pipeline (horizontal, edge-to-edge) ---
    draw_arrow(ax, e1['right'], e1['cy'], e2['left'], e2['cy'])
    draw_arrow(ax, e2['right'], e2['cy'], e3['left'], e3['cy'])
    draw_arrow(ax, e3['right'], e3['cy'], e4['left'], e4['cy'])

    # --- Dynamic object pipeline (horizontal, edge-to-edge) ---
    draw_arrow(ax, e5['right'], e5['cy'], e6['left'], e6['cy'])
    draw_arrow(ax, e6['right'], e6['cy'], e7['left'], e7['cy'])

    # --- Camera image to YOLOv8 (diagonal, bottom-edge to top-edge) ---
    draw_arrow(ax, e1['cx'], e1['bottom'], e5['cx'], e5['top'])

    # --- Dynamic mask feedback: Box7 top -> Box3 bottom (diagonal up) ---
    draw_arrow(ax, e7['cx'], e7['top'], e3['cx'], e3['bottom'],
               color='#E65100', lw=1.8)

    # --- Semantic weight: Box3 bottom -> Box8 top (vertical) ---
    draw_arrow(ax, e3['cx'], e3['bottom'], e8['cx'], e8['top'],
               color='#2E7D32', lw=1.8)

    # --- Pose to Loop Closure: Box4 bottom -> Box9 top (curved, long) ---
    draw_arrow(ax, e4['cx'], e4['bottom'], e9['cx'], e9['top'],
               color='#6A1B9A', lw=1.8,
               connectionstyle='arc3,rad=-0.4')

    # --- Map feedback: Box9 right -> Box2 bottom (curved) ---
    draw_arrow(ax, e9['right'], e9['cy'], e2['cx'], e2['bottom'],
               color='#6A1B9A', lw=1.8,
               connectionstyle='arc3,rad=0.3')

    # ============================================================
    # Arrow labels
    # ============================================================
    label_kw = dict(fontsize=7, fontstyle='italic', color='#455A64', zorder=4,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.85))

    ax.text((e1['right'] + e2['left']) / 2, e1['cy'] + 0.25, 'Features',
            **label_kw)
    ax.text((e2['right'] + e3['left']) / 2, e2['cy'] + 0.25, 'Pose Prior',
            **label_kw)
    ax.text((e3['right'] + e4['left']) / 2, e3['cy'] + 0.25, 'Optimized',
            **label_kw)

    ax.text((e5['right'] + e6['left']) / 2, e5['cy'] + 0.25, 'Detections',
            **label_kw)
    ax.text((e6['right'] + e7['left']) / 2, e6['cy'] + 0.25, 'Verified',
            **label_kw)

    # Diagonal label
    mid_x = (e1['cx'] + e5['cx']) / 2
    mid_y = (e1['bottom'] + e5['top']) / 2
    ax.text(mid_x + 0.15, mid_y, 'RGB', **label_kw)

    # Feedback label
    mid_x_fb = (e7['cx'] + e3['cx']) / 2 + 0.3
    mid_y_fb = (e7['top'] + e3['bottom']) / 2
    ax.text(mid_x_fb, mid_y_fb, 'Dynamic\nMask', fontsize=6.5,
            fontstyle='italic', color='#BF360C', ha='center', va='center',
            zorder=4,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF3E0',
                      edgecolor='none', alpha=0.9))

    # Semantic weight label
    ax.text(e3['cx'] + 0.35, (e3['bottom'] + e8['top']) / 2, 'Semantic\nWeights',
            fontsize=6.5, fontstyle='italic', color='#1B5E20', ha='left', va='center',
            zorder=4,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#E8F5E9',
                      edgecolor='none', alpha=0.9))

    # Loop closure label
    ax.text(7.0, 1.8, 'Loop Closure\n& Global BA', fontsize=6.5,
            fontstyle='italic', color='#4A148C', ha='center', va='center',
            zorder=4,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#F3E5F5',
                      edgecolor='none', alpha=0.9))

    # ============================================================
    # Title
    # ============================================================
    ax.text(7, 7.15, 'System Architecture: Semantic Visual-Inertial SLAM with YOLOv8',
            ha='center', fontsize=13, fontweight='bold', color='#1A237E')

    # ============================================================
    # Legend
    # ============================================================
    legend_patches = [
        mpatches.Patch(color='#BBDEFB', label='SLAM Pipeline'),
        mpatches.Patch(color='#FFCCBC', label='Dynamic Object Handling'),
        mpatches.Patch(color='#C8E6C9', label='Optimization'),
        mpatches.Patch(color='#E1BEE7', label='Map Management'),
    ]
    legend = ax.legend(handles=legend_patches, loc='lower left',
                       fontsize=7, ncol=4, framealpha=0.9,
                       edgecolor='#B0BEC5', fancybox=True,
                       bbox_to_anchor=(0.02, -0.02))
    legend.set_zorder(10)

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(output_dir, 'fig01_system_architecture.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Fig.1: System Architecture - Done')


if __name__ == '__main__':
    generate()