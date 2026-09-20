import json, re, os
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))
labels = ['sitting_static','sitting_xyz','walking_static','walking_xyz','walking_halfsphere']
def num(s):
    m = re.search(r'[0-9]*\.?[0-9]+', s)
    return float(m.group()) if m else None
def load(name):
    txt = open(f"{OUT}/raw_{name}.txt").read().split()
    return [num(x) for x in txt if num(x) is not None]
res = {}
for lbl in labels:
    b, m = load(f"{lbl}_baseline"), load(f"{lbl}_mask")
    row = {'seq': lbl, 'baseline': {'n': len(b)}, 'mask': {'n': len(m)}}
    for k, a in (('baseline', b), ('mask', m)):
        if a:
            row[k]['rmse_mean'] = float(np.mean(a))
            row[k]['rmse_std'] = float(np.std(a, ddof=1)) if len(a) > 1 else None
            row[k]['rmse_pairs'] = a
    if b and m:
        imp = (np.mean(b) - np.mean(m)) / np.mean(b) * 100
        row['improvement'] = {'mean_pct': float(imp),
                              'baseline_mean': float(np.mean(b)), 'mask_mean': float(np.mean(m))}
    res[lbl] = row
with open(f"{OUT}/p1_stats.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
print("== P1 stats saved:", f"{OUT}/p1_stats.json")
