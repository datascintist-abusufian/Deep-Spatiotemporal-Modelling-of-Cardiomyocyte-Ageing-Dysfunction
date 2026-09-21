# Deep Spatiotemporal Modelling of Cardiomyocyte Ageing Dysfunction

Optical-flow-driven detection of cardiomyocyte ageing and injury from live-cell microscopy video, using a Transformer encoder over motion-phenotype time series.

Presented at the British Society for Cardiovascular Research / British Cardiovascular Society meeting: *Advanced transformer-based AI framework for early detection and prediction of cardiomyocyte ageing and injury using motion phenotyping*, Heart 111 (Suppl 3), A269-A271 (2025).

## Motivation

Functional decline in ageing cardiomyocytes shows up in how the cells move before it is obvious in how they look. Frame-to-frame optical flow turns a microscopy video into a time series of contraction and relaxation descriptors, and a sequence model can then separate healthy, aged and damaged cells from that signal alone.

## What is in this repository

`Predictive_Modelling_only_86%.ipynb` - the predictive-modelling stage of the pipeline. It takes a table of per-frame-pair motion features extracted from microscopy video and trains a Transformer classifier to label short temporal windows as Healthy, Aged or Damaged. Upstream optical-flow extraction is run separately; this notebook consumes its output.

## Data

Expected input is `merged_dataset_with_updated_entropy.csv`, one row per frame pair, with the following columns.

- Motion: `Mean_Magnitude`, `Std_Magnitude`, `Max_Magnitude`, `Min_Magnitude`, `Median_Magnitude`, `Mean_Motion_Magnitude`, `Active_Pixel_Ratio`, `Motion_Entropy`, `Direction_Variability`
- Contractile function: `Contraction_Amplitude`, `Relaxation_Time`, `Beat_Frequency`, `Beat_Interval_Variability`, `Peak_Contraction_Velocity`, `Peak_Relaxation_Velocity`
- Morphology and signal quality: `Cell_Morphology_Index`, `Nuclear_Morphology_Index`, `tSNR`
- Index and label: `Frame Pair`, `Classification` (Healthy, Aged, Damaged)

The underlying imaging data are not redistributed here. Please get in touch about access.

## Method

1. Label encoding: Healthy to 0, Aged to 1, Damaged to 2.
2. Scaling: `RobustScaler` across all feature columns, so heavy-tailed motion magnitudes do not dominate.
3. Windowing: sliding windows of 8 consecutive frame pairs, labelled by the final element of the window.
4. Splits: stratified 70 / 15 / 15 train, validation and test, `random_state=42`.
5. Model: linear embedding to d_model 64, a learned positional embedding, a 2-layer `nn.TransformerEncoder` with 4 attention heads and dropout 0.1, attention pooling over the sequence, then a two-layer classification head.
6. Evaluation: accuracy, per-class precision, recall and F1, and a confusion matrix on the held-out test split.

Every run calls `set_seed(42)`, which fixes the Python, NumPy, PyTorch and CUDA seeds and sets `cudnn.deterministic = True`, so the reported figures are reproducible.

## Result

The committed run reaches approximately 86 per cent test accuracy across the three classes. The notebook outputs contain the full classification report and confusion matrix.

## Running it

Developed in Google Colab on an NVIDIA A100 with PyTorch built against CUDA 11.8.

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

Open the notebook, point `load_and_prepare` at your feature CSV and run all cells. A GPU is convenient but not required at this model size.

## Related projects

- [triFuse-pytorch](https://github.com/datascintist-abusufian/triFuse-pytorch) - scribble-supervised cardiac MRI segmentation
- [Cardiomyocyte-cell-motion-analysis](https://github.com/datascintist-abusufian/Cardiomyocyte-cell-motion-analysis) - upstream motion phenotyping
- [Cardiomyocyte-Ageing-Nucleus-Data-Analysis](https://github.com/datascintist-abusufian/Cardiomyocyte-Ageing-Nucleus-Data-Analysis) - nuclear morphology analysis

## Citation

Advanced transformer-based AI framework for early detection and prediction of cardiomyocyte ageing and injury using motion phenotyping. Heart 2025;111(Suppl 3):A269-A271.

## Author

Md Abu Sufian, PhD researcher, School of Architecture, Computing and Engineering, University of East London. [GitHub profile](https://github.com/datascintist-abusufian) | [LinkedIn](https://www.linkedin.com/in/tacticalbusinessintelligence/)
