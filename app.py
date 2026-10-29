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

# --- 1. PAGE CONFIG & CUSTOM CSS (The "Neat" Look) ---
st.set_page_config(page_title="CrackScan | Dashboard", page_icon="🧱", layout="wide")

st.markdown("""
<style>
    /* Force Subtle Background Theme */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f4f6f9 !important; 
    }
    
    /* Main container width */
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Hide default Streamlit sidebar navigation */
    [data-testid="stSidebarNav"] {display: none;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding-top: 2rem;
    }
    
    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 40px; border-radius: 12px;
        margin-bottom: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .hero h1 { color: white; margin: 0; font-size: 2.8rem; font-weight: 700; }
    .hero p { color: #e0e0e0; font-size: 1.15rem; margin-top: 10px; margin-bottom: 0; }
    .hero-highlight { color: #ffd700; } /* Yellow highlight for project name */
    
    /* Verdict Cards */
    .verdict-card {
        padding: 25px; border-radius: 12px; text-align: center;
        margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .verdict-good { background-color: #e6f4ea; border: 2px solid #34a853; }
    .verdict-bad { background-color: #fce8e6; border: 2px solid #ea4335; }
    .verdict-good h2 { color: #1e8e3e; margin: 0; font-size: 2rem; }
    .verdict-bad h2 { color: #d93025; margin: 0; font-size: 2rem; }
    .verdict-card p { margin: 10px 0 0 0; font-size: 1.1rem; color: #333; }
    
    /* Severity Meter */
    .meter-container { margin-top: 15px; text-align: left; }
    .meter-bg {
        width: 100%; height: 12px; background-color: #e0e0e0;
        border-radius: 10px; overflow: hidden; margin-top: 5px;
    }
    .meter-fill-low { height: 100%; width: 33%; background-color: #34a853; }
    .meter-fill-med { height: 100%; width: 66%; background-color: #fbbc04; }
    .meter-fill-high { height: 100%; width: 100%; background-color: #ea4335; }
</style>
""", unsafe_allow_html=True)

HAS_TORCH = importlib.util.find_spec("torch") is not None
CLASSICAL = "Classical CV baseline (no training)"

# ----------------------------------------------------------------------------- helpers
def list_weights(prefix): return sorted(WEIGHTS_DIR.glob(f"{prefix}_*.pt"))
def pretty(path, table):
    key = path.stem.split("_", 1)[1]
    return table.get(key, key)
def load_results(prefix):
    out = []
    for p in sorted(RESULTS_DIR.glob(f"{prefix}_*_metrics.json")):
        try: out.append(json.loads(p.read_text()))
        except Exception: pass
    return out
def load_robustness():
    out = []
    for p in sorted(RESULTS_DIR.glob("robustness_*.json")):
        try: out.append(json.loads(p.read_text()))
        except Exception: pass
    return out

@st.cache_resource(show_spinner="Loading model...")
def get_seg_model(path):
    from src.inference import load_seg_model
    return load_seg_model(path)
@st.cache_resource(show_spinner="Loading classifier...")
def get_cls_model(path):
    from src.inference import load_cls_model
    return load_cls_model(path)

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
    src = st.radio("Image source", ["Upload an image", "Use a sample image"], horizontal=True, key=f"{key}_src")
    if src == "Upload an image":
        f = st.file_uploader("Upload Surface Photo", type=[e.strip(".") for e in IMG_EXT], key=f"{key}_up")
        if f is None: return None, None, None
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
    if mp.exists(): gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
    return limit_size(img), choice, gt

def segment(img, model_choice, use_rec_thr, thr, size):
    if model_choice == CLASSICAL:
        return analysis.classical_crack_mask(img).astype(np.float32), 0.5, "classical baseline"
    from src.inference import predict_seg_prob
    model, ck = get_seg_model(str(WEIGHTS_DIR / model_choice))
    t = float(ck.get("threshold", 0.5)) if use_rec_thr else thr
    return predict_seg_prob(model, img, size), t, f"{ck.get('arch')} | thr={t:.2f}"


# ----------------------------------------------------------------------------- Pages
def page_overview():
    st.title("🏠 Project Overview")
    st.caption("Structural Health Monitoring | SR Gudlavalleru Engineering College | CSE (AI & ML) Final Project")
    st.markdown(
        "Cracks in roads, bridges and buildings are early signs of deterioration. **CrackScan** takes a photo of a "
        "concrete / road surface and tells you **whether a crack is present, where it is, and how severe it looks**."
    )
    
    st.subheader("Objective tracker")
    seg, cls, rob = load_results("seg"), load_results("cls"), load_robustness()
    summary = RESULTS_DIR / "dataset_summary.json"
    has_attn = any(r.get("arch") == "attention_unet" for r in seg)
    rows = [
        ("1", "Study DL approaches for classification, detection, segmentation", "✅ Done"),
        ("2", "Collect and preprocess public crack datasets", "✅ Done" if summary.exists() else "⏳ Pending"),
        ("3", "Implement and compare baseline architectures", "✅ Done" if (len(seg) >= 1 and len(cls) >= 1) else "⏳ Pending"),
        ("4", "One practical enhancement (attention + Dice loss)", "✅ Done" if has_attn else "⏳ Pending"),
        ("5", "Demo interface: upload image -> predictions", "✅ Done"),
        ("6", "Robustness + standard metrics", "✅ Done" if len(rob) >= 1 else "⏳ Pending"),
    ]
    st.dataframe(pd.DataFrame([{"#": a, "Objective": b, "Status": c} for a, b, c in rows]), hide_index=True, use_container_width=True)


def page_inspection():
    # --- UPDATED HERO BANNER ---
    st.markdown("""
    <div class="hero">
        <h1>See what the AI sees.</h1>
        <p>Welcome to <b class="hero-highlight">CrackScan</b>. Upload a surface photo. The app segments cracks, measures them, and draws boxes. <b>Best results:</b> close-up, well-lit photos.</p>
    </div>
    """, unsafe_allow_html=True)

    seg_w, cls_w = list_weights("seg"), list_weights("cls")

    with st.expander("⚙️ Advanced Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        options = [p.name for p in seg_w] if HAS_TORCH else []
        labels = {p.name: pretty(p, SEG_MODELS) for p in seg_w}
        model_choice = c1.selectbox("Model", options + [CLASSICAL], format_func=lambda x: labels.get(x, x))
        deep = model_choice != CLASSICAL
        use_rec = c1.checkbox("Use tuned threshold", True, disabled=not deep)
        thr = c2.slider("Threshold", 0.05, 0.95, 0.5, 0.05, disabled=not deep or use_rec)
        size = c2.select_slider("Inference Size", [256, 320, 384, 512], 256, disabled=not deep)
        min_area = c3.slider("Ignore specks (px)", 0, 500, 30, 5)
        mm = c3.number_input("mm / pixel (0=unknown)", 0.0, 50.0, 0.0, 0.05)
        alpha = c1.slider("Opacity", 0.1, 0.9, 0.55, 0.05)
        show_boxes = c2.checkbox("Show boxes", True)
        show_skel = c3.checkbox("Show skeleton", False)

    with st.container(border=True):
        st.markdown("### 📤 Upload Image")
        img, name, gt = pick_image("insp")
        
    if img is None:
        st.info("👆 Please upload an image or select a sample to begin analysis.")
        return

    prob, t, note = segment(img, model_choice, use_rec, thr, size)
    mask = analysis.clean_mask(prob >= t, min_area)
    stats = analysis.crack_stats(mask, mm or None)
    boxes = analysis.boxes_from_mask(mask) if show_boxes else []
    over = analysis.overlay(img, mask, boxes, alpha, show_skeleton=show_skel)
    found = stats["n_regions"] > 0 and stats["area_pct"] >= 0.02

    sev_text = str(stats['severity']).capitalize()
    if sev_text in ["Low", "Minor"]: meter_class = "meter-fill-low"
    elif sev_text in ["Moderate", "Medium"]: meter_class = "meter-fill-med"
    else: meter_class = "meter-fill-high"

    if found:
        st.markdown(f"""
        <div class="verdict-card verdict-bad">
            <h2>⚠️ Crack Detected</h2>
            <p>Estimated Severity: <b>{sev_text}</b> &nbsp;|&nbsp; Method: {note}</p>
            <div class="meter-container">
                <span style="font-size:0.9rem; color:#555;">Severity Meter: Low ➔ High</span>
                <div class="meter-bg"><div class="{meter_class}"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-card verdict-good">
            <h2>✅ No Significant Crack Detected</h2>
            <p>Method: {note}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crack Area", f"{stats['area_pct']:.2f} %")
    if mm:
        m2.metric("Length", f"{stats['length_mm']:.0f} mm")
        m3.metric("Max Width", f"{stats['max_width_mm']:.2f} mm")
    else:
        m2.metric("Length", f"{stats['length_px']} px")
        m3.metric("Max Width", f"{stats['max_width_px']:.1f} px")
    m4.metric("Regions", stats["n_regions"])

    tabs = st.tabs(["📷 Visual Overlay", "📝 Detailed Data & Models", "📥 Download Report"])
    
    with tabs[0]:
        a, b, d = st.columns(3)
        a.image(img, caption="Original", use_container_width=True)
        b.image((mask * 255).astype(np.uint8), caption="Predicted Mask", use_container_width=True)
        d.image(over, caption="Overlay", use_container_width=True)

    with tabs[1]:
        st.markdown("#### Regions")
        if boxes:
            df = pd.DataFrame(boxes)
            df.index = df.index + 1
            df.columns = ["X", "Y", "Width", "Height", "Pixels"]
            st.dataframe(df, use_container_width=True)
            
        st.markdown("#### Check against Ground Truth")
        gtf = st.file_uploader("Upload Mask for IoU/Dice", type=["png", "jpg", "jpeg", "bmp"], key="gt_up")
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

        # --- YOLO TAB REMOVED, CLASSIFICATION SHOWN DIRECTLY ---
        st.markdown("#### Classification Details")
        if HAS_TORCH and cls_w:
            from src.inference import predict_cls_prob
            rows = []
            for p in cls_w:
                m, ck = get_cls_model(str(p))
                pr = predict_cls_prob(m, img, ck.get("size", 224))
                rows.append({"Model": pretty(p, CLS_MODELS), "P(crack)": round(pr, 3), "Verdict": "Crack" if pr >= 0.5 else "No crack"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No classifier weights found. (Train with `python -m src.train_cls`)")

    with tabs[2]:
        rep = {"image": name, "method": note, **stats}
        st.download_button("⬇️ Download Overlay (PNG)", png_bytes(over), "crack_overlay.png", "image/png")
        st.download_button("⬇️ Download Mask (PNG)", png_bytes((mask * 255).astype(np.uint8)), "crack_mask.png", "image/png")
        st.download_button("⬇️ Download Data (CSV)", pd.DataFrame([rep]).to_csv(index=False), "crack_report.csv", "text/csv")


def page_comparison():
    st.title("📊 Model Comparison")
    seg, cls = load_results("seg"), load_results("cls")
    if not (seg or cls):
        st.info("No result files in `results/` yet.")
        return

    st.subheader("Segmentation (U-Net vs Attention U-Net)")
    if seg:
        df = pd.DataFrame([{"Model": r["label"], "Params (M)": round(r["params"] / 1e6, 2), "IoU": r["test"]["iou"], "Dice": r["test"]["dice"], "Precision": r["test"]["precision"], "Recall": r["test"]["recall"]} for r in seg])
        st.dataframe(df.style.format({c: "{:.3f}" for c in ["IoU", "Dice", "Precision", "Recall"]}), hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Model")[["IoU", "Dice"]])

    st.subheader("Classification")
    if cls:
        dfc = pd.DataFrame([{"Model": r["label"], "Accuracy": r["test"]["accuracy"], "F1": r["test"]["f1"]} for r in cls])
        st.dataframe(dfc.style.format({c: "{:.3f}" for c in ["Accuracy", "F1"]}), hide_index=True, use_container_width=True)
        st.bar_chart(dfc.set_index("Model")[["Accuracy", "F1"]])


def page_robustness():
    st.title("🧪 Robustness & Metrics")
    rob = load_robustness()
    if not rob:
        st.info("No robustness file yet.")
        return
        
    st.subheader("Saved robustness study")
    clean = pd.DataFrame({r["model"]: r["clean"] for r in rob}).T[["dice", "iou"]]
    st.dataframe(clean.style.format("{:.3f}"), use_container_width=True)
    
    kinds = list(rob[0]["perturbed"].keys())
    kind = st.selectbox("Perturbation", kinds)
    chart = pd.DataFrame({r["model"]: [r["clean"]["dice"]] + [r["perturbed"][kind][s]["dice"] for s in "123"] for r in rob}, index=["clean", "sev 1", "sev 2", "sev 3"])
    st.line_chart(chart)


def page_study():
    st.title("📚 Approaches & Datasets")
    st.subheader("Datasets")
    st.dataframe(pd.DataFrame([
        {"Dataset": "DeepCrack", "Task": "Segmentation", "Used for": "Segmentation (main)"},
        {"Dataset": "CRACK500", "Task": "Segmentation", "Used for": "Unseen-surface test"},
        {"Dataset": "SDNET2018", "Task": "Classification", "Used for": "Classification (optional)"},
    ]), hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------- Navigation
with st.sidebar:
    st.markdown("<h2 style='color:#1e3c72; margin-bottom:0;'>🧱 CrackScan</h2>", unsafe_allow_html=True)
    st.caption("Deep Learning for Structural Health")
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
    
    page = st.selectbox("Navigate to:", [
        "🔍 Crack Inspection", 
        "🏠 Project Overview", 
        "📊 Model Comparison", 
        "🧪 Robustness & Metrics", 
        "📚 Approaches & Datasets"
    ], label_visibility="collapsed")

# Route to the correct page
if page == "🔍 Crack Inspection":
    page_inspection()
elif page == "🏠 Project Overview":
    page_overview()
elif page == "📊 Model Comparison":
    page_comparison()
elif page == "🧪 Robustness & Metrics":
    page_robustness()
elif page == "📚 Approaches & Datasets":
    page_study()
