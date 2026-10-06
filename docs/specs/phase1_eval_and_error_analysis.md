# Technical Specification: Phase 1 — Diagnostic Foundation & Error Analysis

**Document Version:** 1.0.0  
**Phase:** Phase 1 (Baseline Evaluation & Visual Error Analysis)  
**Status:** Approved Technical Specification  
**Target Deliverables:** Standalone Interactive HTML Report (`reports/baseline_evaluation/index.html`) + Structured Metrics JSON  

---

## 1. Scope & Objective

This document defines the technical architecture and implementation specifications for **Phase 1**.
The primary objective of this phase is to construct a self-contained evaluation and diagnostic harness for the YOLOv8 model (`deployment/best.pt`) on the real-world validation dataset (32 images in `data/processed/images/val`).

Rather than relying solely on global performance metrics (such as overall mAP), this harness breaks down detection failures into structural error categories:
1. **False Positives (Clutter vs Confusion):** Distinguishing between background hallucinations (trees, generic utility poles, street banners) and cross-provider classification confusion (e.g., CBN pole predicted as MyRepublic).
2. **False Negatives (Missed Objects):** Identifying uncaptured targets, specifically thin objects, distant infrastructure, and minority classes (*Lintasarta*, *Indosat*).
3. **Data-Centric Scale Slicing:** Profiling performance across object area buckets according to COCO standards (*Small*, *Medium*, *Large*).
4. **Interactive Standalone Visual Report:** Delivering an offline-capable HTML dashboard with dynamic SVG bounding box overlays, filterable galleries, and zoom-in inspection modals.

---

## 2. Directory Structure & File Manifest

All Phase 1 code is organized under `scripts/evaluation/`, with output artifacts written to `reports/baseline_evaluation/`:

```text
scripts/
└── evaluation/
    ├── __init__.py
    ├── evaluator.py       # YOLOv8 inference runner & Ground Truth loader
    ├── box_matcher.py     # Greedy IoU matcher: classifies TP, FP_BG, FP_CONF, FN
    ├── slicer.py          # Scale sensitivity (COCO standard) & aspect ratio analysis
    ├── report_builder.py  # Standalone HTML report assembler (embedded CSS + Vanilla JS)
    └── run_evaluation.py  # CLI entry point for full pipeline execution

tests/
└── test_box_matcher.py    # Unit tests validating IoU math and greedy matching logic

reports/
└── baseline_evaluation/
    ├── index.html         # Interactive standalone dashboard report
    ├── metrics.json       # Formal metrics summary (P, R, F1, mAP per class & overall)
    └── error_catalog.json # Granular failure catalog per image and bounding box
```

---

## 3. Data Contracts & Interfaces

### 3.1 Input Contract
* **Dataset Configuration:** `data/processed/data.yaml`
  * Class ID Mapping:
    * `0`: Indihome
    * `1`: Indosat
    * `2`: MyRepublic
    * `3`: Lintasarta
    * `4`: CBN
  * Images path: `data/processed/images/val/*.jpg` (32 images)
  * Labels path: `data/processed/labels/val/*.txt` (YOLO format: `class_id x_center y_center width height` normalized to $[0, 1]$).
* **Model Checkpoint:** `deployment/best.pt`

### 3.2 Output Contract (`metrics.json`)
```json
{
  "metadata": {
    "model_path": "deployment/best.pt",
    "dataset_yaml": "data/processed/data.yaml",
    "imgsz": 1248,
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
    "total_images": 32,
    "timestamp": "2026-10-06T19:00:00Z"
  },
  "overall": {
    "precision": 0.852,
    "recall": 0.781,
    "f1_score": 0.815,
    "map50": 0.763,
    "map50_95": 0.521,
    "total_gt": 54,
    "total_pred": 58,
    "tp": 42,
    "fp_background": 10,
    "fp_confusion": 6,
    "fn": 12
  },
  "per_class": {
    "Lintasarta": {
      "class_id": 3,
      "gt_count": 2,
      "pred_count": 3,
      "tp": 1,
      "fp_background": 1,
      "fp_confusion": 1,
      "fn": 1,
      "precision": 0.333,
      "recall": 0.500,
      "f1_score": 0.400,
      "map50": 0.497,
      "small_sample_warning": true
    }
  },
  "scale_breakdown": {
    "small": { "gt_count": 14, "tp": 8, "fn": 6, "recall": 0.571 },
    "medium": { "gt_count": 28, "tp": 24, "fn": 4, "recall": 0.857 },
    "large": { "gt_count": 12, "tp": 10, "fn": 2, "recall": 0.833 }
  }
}
```

---

## 4. Component Design & Algorithms

### 4.1 Inference & Label Ingestion (`evaluator.py`)
* Executes inference via `ultralytics.YOLO` on `images/val` at high resolution `imgsz=1248` (matching the v2.1 benchmark configuration).
* Converts bounding boxes from normalized $[0, 1]$ coordinates to absolute image pixel coordinates $[x_1, y_1, x_2, y_2]$.
* Returns structured image records:
  * `image_id`: Image filename (e.g. `images_0008.jpg`).
  * `width`, `height`: Native pixel dimensions.
  * `ground_truths`: List of `{box: [x1, y1, x2, y2], class_id: int, area: float}`.
  * `predictions`: List of `{box: [x1, y1, x2, y2], class_id: int, confidence: float, area: float}`.

### 4.2 Greedy IoU Matching Engine (`box_matcher.py`)
Each validation image is evaluated independently:
1. Sort all predictions descending by `confidence`.
2. Compute pairwise IoU matrix between $P$ predictions and $G$ ground truth instances:
   $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$
3. Initialize matching state tracking arrays: `matched_gt = [False] * G`, `matched_pred = [False] * P`.
4. For each prediction $p_i$ (in descending order of confidence):
   * Identify unmatched candidate ground truth $g_j$ (`matched_gt[j] == False`) with maximum IoU.
   * If $\text{IoU}(p_i, g_j) \ge \text{iou\_threshold}$:
     * If $class(p_i) == class(g_j)$:
       * Classify $p_i$ as **`TRUE_POSITIVE`**.
       * Mark `matched_gt[j] = True`.
     * If $class(p_i) \neq class(g_j)$:
       * Classify $p_i$ as **`FP_CONFUSION`** (cross-class provider confusion).
       * Keep `matched_gt[j] = False` (the ground truth target has not been properly detected).
   * If $\text{IoU}(p_i, g_j) < \text{iou\_threshold}$ or no unmatched GT remains:
     * Classify $p_i$ as **`FP_BACKGROUND`** (background clutter hallucination).
5. For all ground truths $g_k$ where `matched_gt[k] == False`:
   * Classify $g_k$ as **`FALSE_NEGATIVE`** (missed target).

### 4.3 Data-Centric Slicing (`slicer.py`)
Categorizes detected and missed instances into diagnostic slices:
* **Object Scale (COCO Benchmark Standards):**
  * $\text{Small}$: $\text{Area} < 32^2 = 1024 \text{ px}^2$.
  * $\text{Medium}$: $1024 \le \text{Area} < 96^2 = 9216 \text{ px}^2$.
  * $\text{Large}$: $\text{Area} \ge 9216 \text{ px}^2$.
* **Geometric Aspect Ratio:**
  * Aspect ratio $AR = \frac{\text{width}}{\text{height}}$.
  * $AR < 0.2$: Slender vertical pole / pipe.
  * $AR > 5.0$: Horizontal overhead cable span.
* **Confidence Error Severity:**
  * High-Confidence FP: Incorrect prediction with $conf \ge 0.60$.
  * Borderline FN: Missed ground truth where candidate detections fall below threshold ($conf < 0.25$).

### 4.4 Standalone HTML Report Builder (`report_builder.py`)
Assembles a single static file `reports/baseline_evaluation/index.html`:
* **Aesthetics & Theme:**
  * Dark palette (Tailwind Slate: `#0f172a` canvas, `#1e293b` cards, `#334155` borders).
  * System typography (Inter / sans-serif stack) for maximum readability.
  * High data-ink ratio with clear metric hierarchy.
* **Interactive Tables & Confusion Matrix:**
  * Confusion matrix grid ($5 \times 6$ including background class).
  * Statistical fragility badges for low-sample classes (*Lintasarta $N=2$* and *Indosat $N=4$*).
* **Dynamic SVG Bounding Box Overlays:**
  * Links directly to local relative image paths (`../../data/processed/images/val/<filename>`).
  * Overlays boxes dynamically using SVG coordinate viewports:
    * Ground Truth: Solid green stroke (`#22c55e`).
    * True Positive: Solid blue stroke (`#3b82f6`).
    * False Positive Clutter: Dashed red stroke (`#ef4444`).
    * False Positive Confusion: Dashed purple stroke (`#a855f7`).
    * False Negative: Dashed/hatched yellow stroke (`#eab308`).
* **Client-Side Interactivity (Vanilla JS, Zero External Libraries):**
  * Layer toggle controls: `[✓] Ground Truth`, `[✓] Predictions`, `[✓] Highlight Errors`.
  * Filter bar: Filter cards by class name and failure type (All, Only False Negatives, Only Small Objects, Perfect Matches).
  * Modal inspection: Zoom in on full-resolution images with exact bounding box dimensions and IoU metrics.

### 4.5 CLI Entry Point (`run_evaluation.py`)
CLI interface supporting customizable thresholds:
```bash
python scripts/evaluation/run_evaluation.py \
  --model deployment/best.pt \
  --data data/processed/data.yaml \
  --imgsz 1248 \
  --conf 0.25 \
  --iou 0.5 \
  --output-dir reports/baseline_evaluation
```

---

## 5. Edge Cases & Resilience

| Edge Case | Scenario | Implementation Guard |
|-----------|----------|----------------------|
| Zero Division | Class with $TP=0$ and $FP=0$ | Set Precision, Recall, and F1 to `0.0` explicitly without throwing exceptions. |
| Background Images | Image with zero ground truth annotations | Predictions are categorized as `FP_BACKGROUND`; recall denominator handled safely. |
| Duplicate Detections | Multiple boxes on a single pole | Highest confidence box matched as TP; subsequent overlapping predictions marked as `FP_BACKGROUND` (duplicate). |
| Out-of-Bounds Boxes | Normalized box coordinates exceeding image bounds | Clamped to pixel boundary $[0, W]$ and $[0, H]$. |
| CUDA Unavailable | Execution on local CPU environment | Automatically falls back to CPU execution cleanly if `torch.cuda.is_available()` is False. |

---

## 6. Verification & Testing Plan

1. **Unit Testing (`tests/test_box_matcher.py`):**
   * Pairwise IoU math calculation across diverse box overlap orientations.
   * Greedy matching verification:
     * 100% overlap with identical class $\to$ TP.
     * High overlap ($IoU=0.8$) with different class $\to$ FP_CONFUSION.
     * Zero overlap ($IoU=0.0$) $\to$ FP_BACKGROUND and FALSE_NEGATIVE.
     * Duplicate predictions on a single GT $\to$ 1 TP and 1 FP.
   * COCO scale categorization thresholds ($<1024$, $1024-9216$, $\ge 9216$).
2. **End-to-End Pipeline Execution:**
   * Run `python scripts/evaluation/run_evaluation.py`.
   * Verify generated artifacts in `reports/baseline_evaluation/`: `index.html` and `metrics.json`.
   * Assert mathematical consistency: $\text{Total TP} + \text{Total FN} = \text{Total Ground Truth Instances}$.
   * Verify offline interactivity of the HTML dashboard in browser.
