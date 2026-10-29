"""CrackScan - Deep Learning-Based Automated Crack Detection for Structural Health Monitoring.

Run locally:   streamlit run app.py
Pages map 1:1 to the project objectives (see the Overview page)."""
import importlib.util
import io
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import analysis, augment, synthetic
from src.config import (CLS_MODELS, IMG_EXT, RESULTS_DIR, SAMPLES_DIR, SEG_MODELS,
                        WEIGHTS_DIR, YOLO_WEIGHTS)
from src.metrics import seg_metrics

st.set_page_config(page_title="CrackScan | Structural Crack Detection", page_icon="🧱", layout="wide")

HAS_TORCH = importlib.util.find_spec("torch") is not None
CLASSICAL = "Classical CV baseline (no training)"


# ----------------------------------------------------------------------------- helpers
def list_weights(prefix):
    return sorted(WEIGHTS_DIR.glob(f"{prefix}_*.pt"))


def pretty(path, table):
    key = path.stem.split("_", 1)[1]
    return table.get(key, key)


def load_results(prefix):
    out = []
    for p in sorted(RESULTS_DIR.glob(f"{prefix}_*_metrics.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def load_robustness():
    out = []
    for p in sorted(RESULTS_DIR.glob("robustness_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


@st.cache_resource(show_spinner="Loading segmentation model...")
def get_seg_model(path):
    from src.inference import load_seg_model
    return load_seg_model(path)


@st.cache_resource(show_spinner="Loading classifier...")
def get_cls_model(path):
    from src.inference import load_cls_model
    return load_cls_model(path)


@st.cache_resource(show_spinner="Loading YOLO detector...")
def get_yolo(path):
    from src.inference import load_yolo
    return load_yolo(path)


def limit_size(img, max_side=1600):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def pick_image(key):
    """Upload / sample chooser. Returns (rgb image, name, ground-truth mask or None)."""
    src = st.radio("Image source", ["Upload an image", "Use a sample image"], horizontal=True, key=f"{key}_src")
    if src == "Upload an image":
        f = st.file_uploader("Concrete / road / wall surface photo", type=[e.strip(".") for e in IMG_EXT], key=f"{key}_up")
        if f is None:
            return None, None, None
        return limit_size(np.array(Image.open(f).convert("RGB"))), f.name, None
    files = sorted(p for p in SAMPLES_DIR.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not files:
        synthetic.generate_samples(SAMPLES_DIR)
        files = sorted(p for p in SAMPLES_DIR.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    choice = st.selectbox("Sample", [p.name for p in files], key=f"{key}_sel")
    path = SAMPLES_DIR / choice
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    gt = None
    mp = SAMPLES_DIR / "masks" / choice
    if mp.exists():
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
    if choice.startswith("synthetic"):
        st.caption("Synthetic demo image. Add real crack photos to `sample_images/` for the review.")
    return limit_size(img), choice, gt


def segment(img, model_choice, use_rec_thr, thr, size):
    """Returns (binary-ready probability map, threshold used, note)."""
    if model_choice == CLASSICAL:
        return analysis.classical_crack_mask(img).astype(np.float32), 0.5, "classical black-hat filter"
    from src.inference import predict_seg_prob
    model, ck = get_seg_model(str(WEIGHTS_DIR / model_choice))
    t = float(ck.get("threshold", 0.5)) if use_rec_thr else thr
    return predict_seg_prob(model, img, size), t, f"{ck.get('arch')} | loss={ck.get('loss')} | thr={t:.2f}"


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.title("🧱 CrackScan: Deep Learning-Based Crack Detection")
    st.caption("Structural Health Monitoring | SR Gudlavalleru Engineering College | CSE (AI & ML) Final Project, Batch A3")
    st.markdown(
        "Cracks in roads, bridges and buildings are early signs of deterioration. **CrackScan** takes a photo of a "
        "concrete / road surface and tells you **whether a crack is present, where it is (box + pixel mask), and how "
        "severe it looks**, using deep learning models that are trained and compared in this project."
    )
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontsize=11];
      A[label="Public datasets\\nDeepCrack / CRACK500 / SDNET2018"]; B[label="Preprocess + augment\\nresize, normalise, flip, shadow"];
      C[label="Baseline models\\nResNet | U-Net | YOLO"]; D[label="Enhancement\\nAttention U-Net + BCE/Dice"];
      E[label="Evaluation\\nIoU, Dice, F1, mAP, robustness"]; F[label="Inspection website\\nupload -> mask + stats"];
      A->B->C->D->E->F }""")

    st.subheader("Objective tracker")
    seg, cls, rob = load_results("seg"), load_results("cls"), load_robustness()
    summary = RESULTS_DIR / "dataset_summary.json"
    has_attn = any(r.get("arch") == "attention_unet" for r in seg)
    rows = [
        ("1", "Study DL approaches for classification, detection, segmentation", "📚 Approaches & Datasets", True),
        ("2", "Collect and preprocess public crack datasets", "📚 Approaches & Datasets", summary.exists()),
        ("3", "Implement and compare baseline architectures", "📊 Model Comparison", len(seg) >= 1 and len(cls) >= 1),
        ("4", "One practical enhancement (attention + Dice loss)", "📊 Model Comparison", has_attn),
        ("5", "Demo interface: upload image -> predictions / masks", "🔍 Crack Inspection", True),
        ("6", "Robustness + standard metrics", "🧪 Robustness & Metrics", len(rob) >= 1),
    ]
    st.dataframe(pd.DataFrame([{"#": a, "Objective": b, "Where in the website": c,
                                "Status": "✅ ready" if ok else "⏳ run training / evaluation to fill"}
                               for a, b, c, ok in rows]), hide_index=True, use_container_width=True)
    if not HAS_TORCH:
        st.warning("PyTorch is not installed here; only the classical baseline can run.")
    elif not list_weights("seg"):
        st.info("No trained weights found in `weights/` yet. The Crack Inspection page works with the classical "
                "baseline until you train the deep models (see README).")


def page_inspection():
    st.title("🔍 Crack Inspection")
    st.markdown("Upload a surface photo. The app segments cracks, measures them and draws boxes. "
                "**Best results**: close-up, well-lit photos where the crack is a few pixels wide or more.")
    seg_w, cls_w = list_weights("seg"), list_weights("cls")

    with st.sidebar:
        st.header("Settings")
        options = [p.name for p in seg_w] if HAS_TORCH else []
        labels = {p.name: pretty(p, SEG_MODELS) for p in seg_w}
        model_choice = st.selectbox("Segmentation model", options + [CLASSICAL],
                                    format_func=lambda x: labels.get(x, x))
        deep = model_choice != CLASSICAL
        use_rec = st.checkbox("Use validation-tuned threshold", True, disabled=not deep)
        thr = st.slider("Crack probability threshold", 0.05, 0.95, 0.5, 0.05, disabled=not deep or use_rec)
        size = st.select_slider("Inference size (px)", [256, 320, 384, 512], 256, disabled=not deep)
        min_area = st.slider("Ignore specks smaller than (px)", 0, 500, 30, 5)
        mm = st.number_input("Calibration: mm per pixel (0 = unknown)", 0.0, 50.0, 0.0, 0.05,
                             help="If you know the real-world scale, lengths/widths are reported in mm.")
        alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.55, 0.05)
        show_boxes = st.checkbox("Show bounding boxes", True)
        show_skel = st.checkbox("Show crack skeleton (length line)", False)

    img, name, gt = pick_image("insp")
    if img is None:
        st.info("⬆️ Upload an image, or switch to a sample image.")
        return

    prob, t, note = segment(img, model_choice, use_rec, thr, size)
    mask = analysis.clean_mask(prob >= t, min_area)
    stats = analysis.crack_stats(mask, mm or None)
    boxes = analysis.boxes_from_mask(mask) if show_boxes else []
    over = analysis.overlay(img, mask, boxes, alpha, show_skeleton=show_skel)
    found = stats["n_regions"] > 0 and stats["area_pct"] >= 0.02

    if found:
        st.error(f"⚠️ **Crack detected**: estimated severity **{stats['severity']}**")
    else:
        st.success("✅ **No significant crack detected**")
    st.caption(f"Method: {note}")

    c = st.columns(4)
    c[0].metric("Crack area", f"{stats['area_pct']:.2f} %")
    if mm:
        c[1].metric("Length (skeleton)", f"{stats['length_mm']:.0f} mm")
        c[2].metric("Max width", f"{stats['max_width_mm']:.2f} mm")
    else:
        c[1].metric("Length (skeleton)", f"{stats['length_px']} px")
        c[2].metric("Max width", f"{stats['max_width_px']:.1f} px")
    c[3].metric("Crack regions", stats["n_regions"])

    a, b, d = st.columns(3)
    a.image(img, caption="1. Original", use_container_width=True)
    b.image((mask * 255).astype(np.uint8), caption="2. Predicted crack mask", use_container_width=True)
    d.image(over, caption="3. Overlay + boxes", use_container_width=True)

    tabs = st.tabs(["Regions", "Classifier", "YOLO detector", "Check against ground truth", "Download report"])
    with tabs[0]:
        if boxes:
            df = pd.DataFrame(boxes)
            df.index = df.index + 1
            df.columns = ["x", "y", "width", "height", "crack pixels"]
            st.dataframe(df, use_container_width=True)
            st.caption("Boxes are derived from the predicted mask by merging nearby fragments.")
        else:
            st.write("No regions to list.")
    with tabs[1]:
        if HAS_TORCH and cls_w:
            from src.inference import predict_cls_prob
            rows = []
            for p in cls_w:
                m, ck = get_cls_model(str(p))
                pr = predict_cls_prob(m, img, ck.get("size", 224))
                rows.append({"Model": pretty(p, CLS_MODELS), "P(crack)": round(pr, 3),
                             "Verdict": "Crack" if pr >= 0.5 else "No crack"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Image-level classification (whole image resized to 224 px). "
                       "Trained on patches, so it is best on close-up images.")
        else:
            st.info("No classifier weights found (`weights/cls_*.pt`). Train with `python -m src.train_cls`.")
    with tabs[2]:
        if HAS_TORCH and YOLO_WEIGHTS.exists() and importlib.util.find_spec("ultralytics"):
            from src.inference import predict_yolo
            dets = predict_yolo(get_yolo(str(YOLO_WEIGHTS)), img)
            yimg = img.copy()
            for dd in dets:
                cv2.rectangle(yimg, (dd["x"], dd["y"]), (dd["x"] + dd["w"], dd["y"] + dd["h"]), (255, 140, 0), 3)
            st.image(yimg, caption=f"YOLO detections: {len(dets)}", use_container_width=True)
            if dets:
                st.dataframe(pd.DataFrame(dets), hide_index=True)
        else:
            st.info("Optional: `weights/yolo_crack.pt` + `pip install ultralytics` enables the YOLOv8 detector here. "
                    "Boxes above are already derived from the segmentation mask.")
    with tabs[3]:
        st.write("Upload the annotated mask (white = crack) to compute IoU / Dice for this image.")
        gtf = st.file_uploader("Ground-truth mask", type=["png", "jpg", "jpeg", "bmp"], key="gt_up")
        gt_arr = None
        if gtf is not None:
            g = np.array(Image.open(gtf).convert("L"))
            gt_arr = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > (0 if g.max() <= 1 else 127)
        elif gt is not None:
            gt_arr = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if gt_arr is not None:
            m = seg_metrics(mask, gt_arr)
            k = st.columns(4)
            k[0].metric("IoU", f"{m['iou']:.3f}"); k[1].metric("Dice", f"{m['dice']:.3f}")
            k[2].metric("Precision", f"{m['precision']:.3f}"); k[3].metric("Recall", f"{m['recall']:.3f}")
    with tabs[4]:
        rep = {"image": name, "method": note, **stats}
        st.download_button("⬇️ Overlay image (PNG)", png_bytes(over), "crack_overlay.png", "image/png")
        st.download_button("⬇️ Mask (PNG)", png_bytes((mask * 255).astype(np.uint8)), "crack_mask.png", "image/png")
        st.download_button("⬇️ Measurements (CSV)", pd.DataFrame([rep]).to_csv(index=False), "crack_report.csv", "text/csv")
        st.caption("Severity is a simple heuristic on crack area (and width, if calibrated). "
                   "It is a screening aid, not a structural engineering assessment.")


def page_comparison():
    st.title("📊 Model Comparison")
    seg, cls = load_results("seg"), load_results("cls")
    det = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("det_*_metrics.json")]
    if not (seg or cls or det):
        st.info("No result files in `results/` yet. Run the training commands from the README / notebook, "
                "then commit the generated `results/*.json` and `weights/*.pt`.")
        return

    st.subheader("Segmentation (U-Net baseline vs Attention U-Net enhancement)")
    if seg:
        df = pd.DataFrame([{"Model": r["label"], "Loss": r["loss"], "Params (M)": round(r["params"] / 1e6, 2),
                            "IoU": r["test"]["iou"], "Dice": r["test"]["dice"], "Precision": r["test"]["precision"],
                            "Recall": r["test"]["recall"], "ms/img": round(r["ms_per_image"], 1),
                            "Test imgs": r["n_test"]} for r in seg])
        st.dataframe(df.style.format({c: "{:.3f}" for c in ["IoU", "Dice", "Precision", "Recall"]}),
                     hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Model")[["IoU", "Dice", "Precision", "Recall"]])
        pick = st.selectbox("Training curves for", [r["label"] for r in seg])
        h = next(r for r in seg if r["label"] == pick)["history"]
        st.line_chart(pd.DataFrame({"train loss": h["train_loss"], "val Dice": h["val_dice"], "val IoU": h["val_iou"]}))
        if len(seg) >= 2:
            base = next((r for r in seg if r["arch"] == "unet"), None)
            enh = next((r for r in seg if r["arch"] == "attention_unet"), None)
            if base and enh:
                dd = enh["test"]["dice"] - base["test"]["dice"]
                st.success(f"Enhancement effect on test Dice: **{dd:+.3f}** "
                           f"(IoU {enh['test']['iou'] - base['test']['iou']:+.3f}).")
    else:
        st.info("No segmentation results yet.")

    st.subheader("Classification (ResNet / EfficientNet / MobileNet)")
    if cls:
        dfc = pd.DataFrame([{"Model": r["label"], "Params (M)": round(r["params"] / 1e6, 2), **{k.capitalize(): r["test"][k] for k in ("accuracy", "precision", "recall", "f1", "auc")},
                             "ms/img": round(r["ms_per_image"], 1)} for r in cls])
        st.dataframe(dfc.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "Auc"]}),
                     hide_index=True, use_container_width=True)
        st.bar_chart(dfc.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])
        pc = st.selectbox("Confusion matrix for", [r["label"] for r in cls])
        cm = next(r for r in cls if r["label"] == pc)["test"]["confusion"]
        st.dataframe(pd.DataFrame(cm, index=["True: no crack", "True: crack"], columns=["Pred: no crack", "Pred: crack"]))
    else:
        st.info("No classification results yet.")

    st.subheader("Detection (YOLOv8)")
    if det:
        st.dataframe(pd.DataFrame([{"Model": r.get("label", "YOLOv8"), "mAP@0.5": r["map50"], "mAP@0.5:0.95": r["map50_95"],
                                    "Precision": r["precision"], "Recall": r["recall"]} for r in det]),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Optional YOLOv8 results not found (`results/det_yolo_metrics.json`). See README, step 6.")


def page_robustness():
    st.title("🧪 Robustness & Metrics")
    rob = load_robustness()
    st.subheader("Saved robustness study (computed on the held-out test split)")
    if rob:
        names = [r["model"] for r in rob]
        clean = pd.DataFrame({r["model"]: r["clean"] for r in rob}).T[["dice", "iou"]]
        st.markdown("**Clean test images**")
        st.dataframe(clean.style.format("{:.3f}"), use_container_width=True)
        kinds = list(rob[0]["perturbed"].keys())
        kind = st.selectbox("Perturbation", kinds, format_func=lambda k: {
            "dark": "Darker image (shadow-like)", "bright": "Over-exposed", "low_contrast": "Low contrast",
            "shadow": "Cast shadow", "blur": "Blur (out of focus)", "noise": "Sensor noise", "jpeg": "JPEG compression"}.get(k, k))
        chart = pd.DataFrame({r["model"]: [r["clean"]["dice"]] + [r["perturbed"][kind][s]["dice"] for s in "123"] for r in rob},
                             index=["clean", "sev 1", "sev 2", "sev 3"])
        st.line_chart(chart)
        st.caption("Dice score as the perturbation gets stronger. Flatter line = more robust.")

        st.markdown("**Thin vs thick cracks** (test images grouped into thirds by mean crack width)")
        rows = []
        for r in rob:
            for g, v in r.get("by_width", {}).items():
                rows.append({"Model": r["model"], "Crack width": g, "Dice": v["dice"], "IoU": v["iou"], "Images": v["n"]})
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Dice": "{:.3f}", "IoU": "{:.3f}"}), hide_index=True, use_container_width=True)
        cross = [{"Model": r["model"], "Unseen dataset": r["cross_dataset"]["csv"], "Dice": r["cross_dataset"]["dice"],
                  "IoU": r["cross_dataset"]["iou"]} for r in rob if r.get("cross_dataset")]
        if cross:
            st.markdown("**Unseen surfaces** (trained on one dataset, tested on another)")
            st.dataframe(pd.DataFrame(cross), hide_index=True, use_container_width=True)
    else:
        st.info("No robustness file yet. Run `python -m src.evaluate --models unet attention_unet`.")

    st.divider()
    st.subheader("Live robustness test")
    st.write("Apply a perturbation to any image and watch the predicted mask change.")
    img, _, _ = pick_image("rob")
    if img is None:
        return
    c = st.columns(3)
    kind = c[0].selectbox("Perturbation ", augment.PERTURBATIONS, key="live_kind")
    sev = c[1].select_slider("Severity", [1, 2, 3], 2)
    seg_w = list_weights("seg") if HAS_TORCH else []
    model_choice = c[2].selectbox("Model ", [p.name for p in seg_w] + [CLASSICAL], key="live_model")
    pert = augment.perturb_image(img, kind, sev)
    m0, t0, _ = segment(img, model_choice, True, 0.5, 256)
    m1, t1, _ = segment(pert, model_choice, True, 0.5, 256)
    k0, k1 = analysis.clean_mask(m0 >= t0, 30), analysis.clean_mask(m1 >= t1, 30)
    s0, s1 = analysis.crack_stats(k0), analysis.crack_stats(k1)
    x = st.columns(4)
    x[0].image(analysis.overlay(img, k0), caption=f"Original | area {s0['area_pct']:.2f}%", use_container_width=True)
    x[1].image(pert, caption=f"Perturbed: {kind} {sev}", use_container_width=True)
    x[2].image(analysis.overlay(pert, k1), caption=f"Prediction | area {s1['area_pct']:.2f}%", use_container_width=True)
    agree = seg_metrics(k1, k0)
    x[3].metric("Agreement with original prediction (Dice)", f"{agree['dice']:.3f}")


def page_study():
    st.title("📚 Approaches & Datasets")
    st.subheader("1. Deep learning approaches for crack analysis")
    st.dataframe(pd.DataFrame([
        {"Task": "Classification", "Question answered": "Is there a crack in this image / patch?",
         "Models used here": "ResNet-18 (baseline), EfficientNet-B0, MobileNetV3-Small (lightweight)",
         "Output": "P(crack)", "Metrics": "Accuracy, Precision, Recall, F1, AUC",
         "Strength / limit": "Fast and simple; no location information"},
        {"Task": "Object detection", "Question answered": "Where are the cracks (boxes)?",
         "Models used here": "YOLOv8 (optional) and mask-derived boxes",
         "Output": "Bounding boxes", "Metrics": "Precision, Recall, mAP@0.5",
         "Strength / limit": "Good for localisation; boxes are loose for long thin cracks"},
        {"Task": "Semantic segmentation", "Question answered": "Which exact pixels are crack?",
         "Models used here": "U-Net (baseline), Attention U-Net (enhanced)",
         "Output": "Pixel mask -> area, length, width", "Metrics": "IoU, Dice, Precision, Recall",
         "Strength / limit": "Enables measurement; needs pixel-level labels"},
    ]), hide_index=True, use_container_width=True)
    st.markdown(
        "**Related work used as guidance**: *DeepCrack* (Zou et al., IEEE TIP 2019) for hierarchical features and the "
        "DeepCrack dataset; *Efficient Dataset Collection for Concrete Crack Detection with Spatial-Adaptive Data "
        "Augmentation* (Jing et al., IEEE Access 2024) for augmentation; *Complex Crack Segmentation and Quantitative "
        "Evaluation* (Kim & Lee, IEEE Access 2023) for evaluation across thin / complex cracks.")

    st.subheader("Our enhancement (Objective 4)")
    st.markdown(
        "- **Attention gates** on every U-Net skip connection: the decoder tells the skip features which regions matter, "
        "which suppresses background texture and helps keep thin, low-contrast cracks.\n"
        "- **BCE + Dice loss**: cracks are ~1-3 % of pixels, so plain BCE can predict 'all background'. "
        "Dice directly rewards overlap with the thin foreground. A Focal + Dice variant is also available.")

    st.subheader("2. Datasets and preprocessing")
    st.dataframe(pd.DataFrame([
        {"Dataset": "DeepCrack", "Task": "Segmentation", "Contents": "537 RGB images with pixel masks, several scenes/scales", "Used for": "Segmentation (main)"},
        {"Dataset": "CRACK500", "Task": "Segmentation", "Contents": "Pavement images with pixel masks (patches)", "Used for": "Unseen-surface test"},
        {"Dataset": "SDNET2018", "Task": "Classification", "Contents": "56k crack / no-crack images of decks, pavements, walls", "Used for": "Classification (optional)"},
        {"Dataset": "Patches from DeepCrack", "Task": "Classification", "Contents": "128 px crops labelled by the mask", "Used for": "Classification (default, no extra download)"},
    ]), hide_index=True, use_container_width=True)
    st.markdown("**Pipeline**: read -> resize (256 px, area interpolation so thin cracks survive) -> ImageNet normalisation "
                "-> augmentation (flip, rot90, zoom-crop, brightness/contrast, random shadow, blur, noise) "
                "-> 70/15/15 train/val/test split (classification: stratified; patches inherit their source image's split).")
    ds = RESULTS_DIR / "dataset_summary.json"
    if ds.exists():
        st.markdown("**Prepared in this project**")
        st.json(json.loads(ds.read_text()))


def page_about():
    st.title("ℹ️ How it works & limitations")
    st.markdown("""
**Using the website**: go to *Crack Inspection*, upload a photo, read the verdict banner, the three panels
(original / mask / overlay) and the measurements. Use *Check against ground truth* if you have an annotated mask.

**Pipeline per image**: resize -> model gives a crack-probability for each pixel -> threshold -> remove tiny specks
-> skeleton (centre-line) gives length, distance transform gives width -> connected groups become boxes ->
heuristic severity.

**Limitations (be upfront at the review)**
- Trained on public datasets; unseen surface types / lighting can lower accuracy (see *Robustness*).
- Very large photos are resized; hairline cracks may disappear. Take close-ups.
- Lengths/widths are in pixels unless you enter the mm-per-pixel scale.
- Severity is a screening heuristic, not a structural safety certification.

**Next phase**: mobile/desktop application, video/drone input, tiling for large images, digital-twin export.
""")
    st.caption("Team: B. Vinodhini, Ch. Akhila, G. Balaji, K. Lakshmi Gayathri | Guide: Mrs. G. Trinisha, M.Tech")


# ----------------------------------------------------------------------------- Navigation
import streamlit as st

st.sidebar.title("CrackScan Menu")
st.sidebar.markdown("Deep Learning for Structural Health Monitoring")
st.sidebar.divider()

# Section 1: The actual App
st.sidebar.subheader("🚀 The Application")
nav_app = st.sidebar.radio("", ["🔍 Crack Inspection", "ℹ️ How it works"], label_visibility="collapsed")

st.sidebar.divider()

# Section 2: The Research / Guide Demo
st.sidebar.subheader("📊 Research & Evaluation")
nav_research = st.sidebar.radio("", ["🏠 Project Overview", "📊 Model Comparison", "🧪 Robustness & Metrics", "📚 Approaches & Datasets"], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit | SR Gudlavalleru Engineering College")

# Route to the correct page based on the selection
if nav_app == "🔍 Crack Inspection":
    page_inspection()
elif nav_app == "ℹ️ How it works":
    page_about()
elif nav_research == "🏠 Project Overview":
    page_overview()
elif nav_research == "📊 Model Comparison":
    page_comparison()
elif nav_research == "🧪 Robustness & Metrics":
    page_robustness()
elif nav_research == "📚 Approaches & Datasets":
    page_study()