"""
Inference-only Gradio app. Loads model_bundle.joblib (made by export_model.py)
and assigns a drawn/uploaded Tamil character to a learned cluster.
Files needed in the Space: app.py, cluster.py, model_bundle.joblib,
model_bundle_tsne.png (optional), requirements.txt, README.md
"""
import os

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cluster import extract_hog_features, extract_pixel_features

# ZeroGPU Spaces refuse to start without a @spaces.GPU function; harmless no-op.
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return 0
except ImportError:
    pass

BUNDLE = joblib.load("model_bundle.joblib")
TSNE_PNG = "model_bundle_tsne.png" if os.path.exists("model_bundle_tsne.png") else None


def preprocess(img, invert):
    if isinstance(img, dict):            # Sketchpad output
        img = img.get("composite")
    if img is None:
        return None
    im = Image.fromarray(np.asarray(img).astype(np.uint8))
    if im.mode == "RGBA":                # transparent canvas -> white
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    arr = np.array(im.convert("L").resize((64, 64)))
    if invert:
        arr = 255 - arr
    return (arr > 127).astype(np.float32)


def samples_fig(cid, title=None):
    imgs = BUNDLE["samples"][int(cid)]
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, im in zip(axes.flat, imgs):
        ax.imshow(im, cmap="gray")
    fig.suptitle(title or f"Cluster {int(cid)}")
    fig.tight_layout()
    return fig


def predict(sketch, upload, invert):
    src = upload if upload is not None else sketch
    arr = preprocess(src, invert)
    if arr is None:
        raise gr.Error("Draw or upload a character first.")
    feat = extract_hog_features if BUNDLE["features"] == "hog" else extract_pixel_features
    z = BUNDLE["pca"].transform(BUNDLE["scaler"].transform(feat(arr[None])))
    proba = BUNDLE["knn"].predict_proba(z)[0]
    order = np.argsort(proba)[::-1][:3]
    cids = BUNDLE["knn"].classes_
    rows = [{"cluster": int(cids[i]),
             "likely_class": BUNDLE["cluster_to_class"][int(cids[i])],
             "confidence": round(float(proba[i]), 3)} for i in order if proba[i] > 0]
    best = int(cids[order[0]])
    return arr, pd.DataFrame(rows), samples_fig(best, f"Closest cluster: {best}")


def show_cluster(cid):
    cid = int(cid)
    return samples_fig(cid, f"Cluster {cid}  (likely class: {BUNDLE['cluster_to_class'][cid]})")


with gr.Blocks(title="Tamil Handwritten Character Clustering") as demo:
    gr.Markdown(
        "# Tamil Handwritten Character Clustering\n"
        f"Pretrained: **{BUNDLE['algo']}**, {BUNDLE['features'].upper()} features, "
        f"PCA {BUNDLE['pca_dim']}, k = {BUNDLE['k']}, {BUNDLE['n_images']} training images.")
    with gr.Tab("Try a character"):
        gr.Markdown("Draw or upload one glyph. Tick *Invert* if your dataset is white-on-black.")
        with gr.Row():
            sketch = gr.Sketchpad(label="Draw", type="numpy", canvas_size=(256, 256))
            upload = gr.Image(label="...or upload", type="numpy", image_mode="L")
        invert = gr.Checkbox(label="Invert colours", value=False)
        go = gr.Button("Assign to cluster", variant="primary")
        with gr.Row():
            seen = gr.Image(label="What the model sees (64x64)", image_mode="L")
            top3 = gr.Dataframe(label="Closest clusters")
        neigh = gr.Plot(label="Examples from the closest cluster")
        go.click(predict, [sketch, upload, invert], [seen, top3, neigh])
    with gr.Tab("Model results"):
        gr.Dataframe(pd.DataFrame([{"metric": m, "value": round(v, 4)}
                                   for m, v in BUNDLE["metrics"].items()]), label="Metrics")
        if TSNE_PNG:
            gr.Image(TSNE_PNG, label="t-SNE (ground truth vs predicted clusters)")
        cs = [int(c) for c in BUNDLE["cids"]]
        cid = gr.Slider(min(cs), max(cs), value=cs[0], step=1, label="Browse cluster")
        grid = gr.Plot(value=show_cluster(cs[0]))
        cid.release(show_cluster, cid, grid)

if __name__ == "__main__":
    demo.launch()
