# aal_roi_pmap_plotly.py
# Interactive whole-brain AAL, ROI-level coloring by p-value (Plotly Mesh3D).
# Output: a single HTML you can double-click. Hover any ROI to see name + p.

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from skimage import measure
import plotly.graph_objects as go


# ===================== USER INPUT =====================
CSV_PATH = r"C:\Users\USER\Desktop\לימודים\רפואה\מעבדה\KPE\new_data\t_test_ses_1_3\between_group_followup_minus_baseline_amygdala.csv"
ROI_COL  = "region"      # e.g. "region" / "label_name" / "ROI"
PV_COL   = "p_value"     # column with p-values
OUT_HTML = "aal_roi_pmap.html"

# Optional tweaks
STEP_SIZE = 2            # marching-cubes decimation (1 = highest detail)
OPACITY   = 0.85
COLORMAP  = "Turbo"      # Plotly colorscale name (e.g., "Turbo", "Viridis", "RdBu")
# ======================================================

EPS = 1e-16  # to avoid log(0)


def norm_name(n: str) -> str:
    """Normalize ROI names to AAL style with underscores and _L/_R."""
    n = str(n).strip()
    n = n.replace(" Left", "_L").replace(" Right", "_R")
    n = n.replace(" ", "_").replace("-", "_")
    return n


def load_aal_spm12():
    """Return atlas image, name->atlas_value mapping, and label list."""
    print("Fetching AAL atlas (SPM12)...")
    aal = datasets.fetch_atlas_aal(version="SPM12")  # MNI152 2mm labels
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)

    # labels and (crucially) indices = actual voxel values in this atlas (e.g., 2001..9170)
    labels_raw = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    if "indices" in aal:
        indices_raw = list(aal["indices"])
    else:
        # Fallback: derive from unique values (rarely needed; provided here for safety)
        uvals = sorted(v for v in np.unique(atlas) if v > 0)
        indices_raw = uvals[:len(labels_raw)]

    labels_norm = [norm_name(x) for x in labels_raw]
    name2val = {labels_norm[i]: int(indices_raw[i]) for i in range(len(labels_norm))}

    # quick atlas sanity
    u = np.unique(atlas)
    u = u[u > 0]
    print(f"Atlas shape: {atlas.shape} | nonzero labels: {len(u)} | min={u.min()} max={u.max()}")

    return atlas_img, atlas, name2val, labels_norm


def read_csv_pvalues(csv_path: str, roi_col: str, pv_col: str):
    """Read CSV, normalize ROI names, keep smallest p per ROI."""
    print("Reading CSV...")
    df = pd.read_csv(csv_path)

    # auto-detect ROI column if needed
    if roi_col not in df.columns:
        for c in df.columns:
            if c.lower() in {"region", "label", "label_name", "roi", "roi_name"}:
                roi_col = c
                break
    assert roi_col in df.columns, f"ROI column not found. Columns: {df.columns.tolist()}"
    assert pv_col in df.columns, f"P-value column '{pv_col}' not found. Columns: {df.columns.tolist()}"

    df[pv_col] = pd.to_numeric(df[pv_col], errors="coerce")
    df = df.dropna(subset=[pv_col])

    df["_roi_norm"] = df[roi_col].map(norm_name)
    # collapse duplicates by taking the smallest p per ROI
    g = df.groupby("_roi_norm", as_index=False)[pv_col].min()

    return g, roi_col, pv_col


def roi_mesh(atlas_img, atlas_data, atlas_value, step=2):
    """
    Build a surface mesh for voxels where atlas == atlas_value.
    Returns (verts_world, faces, voxel_count). If empty, verts = faces = None.
    """
    mask = (atlas_data == atlas_value).astype(np.uint8)
    vox = int(mask.sum())
    if vox == 0:
        return None, None, 0
    try:
        verts, faces, _, _ = measure.marching_cubes(
            volume=mask,
            level=0.5,
            step_size=step,
            allow_degenerate=True
        )
    except TypeError:
        # older scikit-image signature
        verts, faces, _, _ = measure.marching_cubes(mask, 0.5, step_size=step)
    except Exception:
        # retry with denser sampling
        try:
            verts, faces, _, _ = measure.marching_cubes(mask, 0.5, step_size=1, allow_degenerate=True)
        except Exception:
            return None, None, vox

    # transform voxel coords to world (MNI) coords
    verts_world = nib.affines.apply_affine(atlas_img.affine, verts)
    return verts_world, faces, vox


def main():
    atlas_img, atlas, name2val, labels_norm = load_aal_spm12()
    g, roi_col, pv_col = read_csv_pvalues(CSV_PATH, ROI_COL, PV_COL)

    # map CSV ROIs to atlas voxel values
    rows = []
    missing = []
    for _, r in g.iterrows():
        roi_norm = r["_roi_norm"]
        p = float(r[pv_col])
        val = name2val.get(roi_norm)
        if val is None:
            # also try spaced variant (rarely needed, but harmless)
            spaced = roi_norm.replace("_L", " Left").replace("_R", " Right")
            val = name2val.get(norm_name(spaced))
        if val is None:
            missing.append(roi_norm)
            continue
        rows.append((val, roi_norm, p))

    print(f"ROIs with p-values (after dedupe): {len(rows)}  |  Not matched to AAL: {len(missing)}")
    if missing:
        print("Examples not matched:", missing[:10])

    # preview a few voxel counts so you can see they are >0 now
    preview = []
    for (val, roi_name, p) in rows[:10]:
        mcount = int((atlas == val).sum())
        preview.append((roi_name, val, mcount))
    print("First few ROI voxel counts:", preview)

    # intensity = -log10(p)
    intensities = [max(-np.log10(p + EPS), 0.0) for (_, _, p) in rows]
    if len(intensities) == 0:
        raise RuntimeError("No ROI names matched AAL. Check your CSV ROI names.")
    vmin, vmax = 0.0, max(intensities) if intensities else 1.0

    print("Building meshes & figure...")
    fig = go.Figure()
    first = True
    added = 0
    nonempty = 0

    for (val, roi_name, p), inten in zip(rows, intensities):
        verts, faces, vox = roi_mesh(atlas_img, atlas, val, step=STEP_SIZE)
        if vox > 0:
            nonempty += 1
        if verts is None or faces is None:
            continue

        inten_arr = np.full(verts.shape[0], inten, dtype=float)

        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=inten_arr,
            cmin=vmin, cmax=vmax,
            colorscale=COLORMAP,
            showscale=first,                  # one shared colorbar
            colorbar=dict(title="-log10(p)"),
            opacity=OPACITY,
            flatshading=True,
            name=roi_name,
            hovertemplate=(
                f"<b>{roi_name}</b><br>"
                f"p = {p:.3g}<br>"
                "-log10(p) = %{intensity:.2f}<extra></extra>"
            )
        ))
        first = False
        added += 1

    print(f"ROIs with >0 voxels: {nonempty}")
    print(f"Added {added} ROI meshes.")

    fig.update_layout(
        title="AAL ROI map colored by p-value (lower p = hotter color)",
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
        ),
        paper_bgcolor="white",
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True, auto_open=False)
    print(f"\nWrote: {OUT_HTML}\nDouble-click it to open (no server needed).")


if __name__ == "__main__":
    main()
