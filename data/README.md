# Dataset

Dataset images and labels are **not committed to this repository** (too large for git).

## Recommended Datasets

| Dataset | Keypoints | License | Link |
|---------|-----------|---------|------|
| FreiHAND | 21 (MediaPipe order) | CC BY 4.0 | https://lmb.informatik.uni-freiburg.de/projects/freihand/ |
| HaGRID | 21 | MIT | https://github.com/hukenovs/hagrid |
| COCO WholeBody (hands) | 21 | CC BY 4.0 | https://github.com/jin-s13/COCO-WholeBody |
| OneHand10K | 21 | research only | https://www.yangangwang.com/papers/WANG-MRF-2018-07.html |

## ⚠️ Keypoint Re-ordering

Most public datasets use **MediaPipe ordering** (thumb-first):
```
MediaPipe:  0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
This model: 0=wrist, 1-4=index, 5-8=middle, 9-12=ring, 13-16=pinky, 17-20=thumb
```

If your source dataset uses MediaPipe ordering, re-map with:

```python
import numpy as np

# MediaPipe -> index-first re-ordering
MEDIAPIPE_TO_MODEL = [0, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 1,2,3,4]

def remap_keypoints(kpts: np.ndarray) -> np.ndarray:
    """kpts shape: [21, 3] (x, y, visibility)"""
    return kpts[MEDIAPIPE_TO_MODEL]
```

## Directory Layout

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Annotation files follow YOLO keypoint format — one `.txt` per image:
```
<class> <cx> <cy> <w> <h>  <kx0> <ky0> <kv0>  ...  <kx20> <ky20> <kv20>
```
All coordinates normalized to `[0, 1]`. Visibility: `0` = not labeled, `1` = labeled, `2` = labeled + visible.
