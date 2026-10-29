# CrackScan: Deep Learning-Based Automated Crack Detection (Streamlit website)

Final-year project, CSE (AI & ML), SR Gudlavalleru Engineering College. The website takes a photo of a concrete / road
surface and reports **crack present?**, a **pixel mask**, **boxes**, and **measurements** (area %, length, width, severity).

## What implements which objective

| # | Objective | Code | Website page |
|---|-----------|------|--------------|
| 1 | Study classification / detection / segmentation approaches | - | Approaches & Datasets |
| 2 | Collect + preprocess public datasets | `src/data_prep.py`, `src/datasets.py`, `src/augment.py` | Approaches & Datasets |
| 3 | Baselines: ResNet/EfficientNet, YOLOv8, U-Net | `src/models.py`, `train_cls.py`, `train_seg.py`, `make_yolo_dataset.py` | Model Comparison |
| 4 | One enhancement | Attention gates + BCE+Dice loss (`models.py`, `losses.py`) | Model Comparison |
| 5 | Demo interface (upload -> masks) | `app.py`, `src/inference.py`, `src/analysis.py` | Crack Inspection |
| 6 | Robustness + metrics | `src/evaluate.py`, `src/metrics.py` | Robustness & Metrics |

## Folder structure
```
crack_detection/
├── app.py                  # Streamlit website (entry point)
├── requirements.txt        # for Streamlit Cloud / local (CPU torch)
├── notebooks/run_on_colab_kaggle.ipynb   # data -> train -> evaluate -> run site
├── src/
│   ├── config.py  data_prep.py  datasets.py  augment.py
│   ├── models.py  losses.py  engine.py  metrics.py
│   ├── train_seg.py  train_cls.py  evaluate.py  make_yolo_dataset.py
│   ├── inference.py  analysis.py  synthetic.py  export_weights.py
├── weights/    # seg_*.pt, cls_*.pt (created by training; commit these)
├── results/    # *_metrics.json, robustness_*.json (created by training; commit these)
├── sample_images/   # demo images (replace the synthetic ones with REAL crack photos)
└── data/       # datasets (not committed)
```

## A. Run on Google Colab or Kaggle (training + evaluation)
1. Zip the `crack_detection` folder as `crack_detection.zip`.
2. Open `notebooks/run_on_colab_kaggle.ipynb` in Colab (T4 GPU) or Kaggle (GPU + Internet ON).
3. Run all cells in order: data -> U-Net vs Attention U-Net -> ResNet-18 vs EfficientNet-B0 -> robustness -> package.
4. Download `crack_detection_trained.zip` (weights) and `crack_detection_results.zip` (metrics) and unzip them into
   the project's `weights/` and `results/` folders on your laptop.
5. The last notebook cell serves the website with a temporary public link if you want to test it there.

Equivalent commands (from the project root):
```bash
python -m src.data_prep seg --root data/DeepCrack
python -m src.data_prep cls-from-seg --seg_csv data/seg_split.csv
python -m src.train_seg --model unet --loss bce
python -m src.train_seg --model attention_unet --loss bce_dice
python -m src.train_cls --model resnet18
python -m src.train_cls --model efficientnet_b0
python -m src.evaluate --models unet attention_unet
python -m src.export_weights        # fp16, each file < 25 MB
```

## B. Run the website locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Without trained weights the site still works with a **classical (non-learned) baseline**; with weights in `weights/` the
deep models appear in the sidebar automatically.

## C. Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repo and push the project **including `weights/` and `results/`** (all files < 25 MB after
   `export_weights`, so the GitHub web "Upload files" button works too; no `data/` needed).
2. Go to share.streamlit.io -> **Create app** -> pick the repo, branch, main file `app.py`.
3. Advanced settings -> Python 3.11 or 3.12. Deploy. First build takes a few minutes (installs CPU torch).
4. Open the link on your phone and laptop and test an upload before the review.

## D. Review-day demo flow (3-4 minutes)
1. **Overview**: show the objective tracker (all six green) and the pipeline diagram.
2. **Approaches & Datasets**: three task types, datasets, preprocessing, the enhancement idea.
3. **Crack Inspection**: upload a real crack photo -> verdict, mask, overlay, measurements, boxes; then an image *without* a crack.
4. **Model Comparison**: baseline vs enhanced numbers and training curves (state honestly what improved and by how much).
5. **Robustness**: line chart under shadow / blur / low contrast, thin-vs-thick table, and the live perturbation tester.
6. **How it works**: limitations and next phase (mobile / desktop app, drone video, tiling for large images).

## Notes and honest limitations
- `src/synthetic.py` makes fake cracks **only for smoke tests**; never report numbers from synthetic data.
- Bounding boxes are derived from predicted masks (YOLOv8 is optional); lengths/widths are in pixels unless a scale is entered.
- Severity is a screening heuristic, not an engineering assessment.
- DeepCrack is licensed for non-commercial research / education; cite Liu et al., *Neurocomputing* 2019.
