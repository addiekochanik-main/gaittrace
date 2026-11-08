"""Export UCI HAR walking-window PCA coordinates plus real walking traces for the website.
Run from repo root: python export_pca.py
"""
from pathlib import Path
import json
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from features import extract_features

DATA = ROOT / 'data' / 'UCI HAR Dataset' / 'train'
NAMES = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
activities = np.loadtxt(DATA / 'y_train.txt', dtype=int)
subjects = np.loadtxt(DATA / 'subject_train.txt', dtype=int)
walking = activities == 1
signals = np.stack([np.loadtxt(DATA / 'Inertial Signals' / f'{name}_train.txt')[walking] for name in NAMES], axis=1)
ids = subjects[walking]

features = extract_features(signals)
scaler = StandardScaler().fit(features)
pca = PCA(n_components=3, random_state=42).fit(scaler.transform(features))
base = pca.transform(scaler.transform(features))
scale = 4.2 / max(float(np.percentile(np.linalg.norm(base, axis=1), 95)), 1e-8)

modified = []
for percentage in range(0, 51, 5):
    altered = signals.copy()
    altered[:, :3, :] *= 1 + percentage / 100
    coords = pca.transform(scaler.transform(extract_features(altered)))
    modified.append(np.round(coords * scale, 4).tolist())

# Real walking windows (6 axes x 128 samples at 50 Hz) for the signal viewer
traces = []
for pid in (27, 17, 1, 7):
    idx = np.where(ids == pid)[0]
    if len(idx) == 0:
        continue
    i = int(idx[len(idx) // 2])
    traces.append({'participant': pid, 'index': i, 'signal': np.round(signals[i], 4).tolist()})

output = {
    'base': np.round(base * scale, 4).tolist(),
    'modified': modified,
    'participants': ids.astype(int).tolist(),
    'traces': traces,
    'hz': 50,
    'metadata': {'dataset': 'UCI HAR train WALKING', 'method': 'PCA of standardized 24 features',
                 'explained_variance_ratio': pca.explained_variance_ratio_.round(5).tolist()},
}
(ROOT / 'pca_data.json').write_text(json.dumps(output, separators=(',', ':')), encoding='utf-8')
print(f'Exported {len(base)} walking windows and {len(traces)} traces to pca_data.json')