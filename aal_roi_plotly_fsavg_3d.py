# aal_roi_plotly_fsavg_3d_seed.py
# Interactive 3D fsaverage cortex colored by AAL ROI p-values (ROI-level).
# - Hover shows the EXACT ROI name from your CSV.
# - Optional seed filtering (e.g., only connections from Amygdala_L/R).
# - Title includes the chosen seed (original CSV label) and p-threshold.
# - Offline HTML (double-click) + optional PNG export.

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface
import plotly.graph_objects as go

# ========================= USER INPUT =========================
CSV_PATH   = r"C:\Users\USER\Desktop\לימודים\רפואה\מעבדה\KPE\new_data\t_test_ses_1_3\between_group_followup_minus_baseline_amygdala.csv"
ROI_COL    = "region"      # target ROI column (e.g., "region" / "label_name" / "ROI")
PV_COL     = "p_value"     # p-value column
SEED_COL   = "seed"          # set to your seed column name if you have one (e.g., "seed"); else None
FILTER_BY_SEED = "Amygdala_L"       # e.g., "Amygdala_L" or "Amygdala_R"; or None to include all

P_THRESH   = None          # show only ROIs with p < P_THRESH; set None to show all
COLORSCALE = "Amp"       # "Magma","Viridis","Turbo","Plasma","Inferno","Cividis", ...
OPACITY    = 1.0           # 0.9–1.0 looks crisp
HEMISPHERE_SHIFT = 55      # mm separation between hemispheres
SHOW_P_IN_HOVER = True     # True: show ROI name + p; False: show name only

OUT_HTML   = "fsavg_roi_pmap_3d.html"
SAVE_PNG   = False         # set True if you want a PNG (requires 'pip install kaleido')
PNG_PATH   = "fsavg_roi_pmap_3d.png"
# =============================================================

EPS = 1e-16

def norm_name(n: str) -> str:
    n = str(n).strip()
    n = n.replace(" Left", "_L").replace(" Right", "_R")
    n = n.replace(" ", "_").replace("-", "_")
    return n

def main():
    print("Fetching AAL (SPM12)...")
    aal = datasets.fetch_atlas_aal(version="SPM12")     # MNI152 2mm
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)

    labels_raw  = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    indices_raw = list(aal["indices"]) if "indices" in aal else sorted(int(v) for v in np.unique(atlas) if v>0)[:len(labels_raw)]
    labels_norm = [norm_name(x) for x in labels_raw]
    # normalized AAL label -> voxel value in atlas (e.g., 2001..9170)
    name2val = {labels_norm[i]: int(indices_raw[i]) for i in range(len(labels_norm))}

    u = np.unique(atlas); u = u[u>0]
    print(f"Atlas shape: {atlas.shape} | labels: {len(u)} | min={u.min()} max={u.max()}")

    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    # -------- use LOCAL copies to avoid UnboundLocalError --------
    roi_col  = ROI_COL
    pv_col   = PV_COL
    seed_col = SEED_COL
    # -------------------------------------------------------------

    # auto-detect ROI column if needed
    if roi_col not in df.columns:
        for c in df.columns:
            if c.lower() in {"region","label","label_name","roi","roi_name"}:
                roi_col = c; break
    assert roi_col in df.columns, f"ROI column not found. Got: {df.columns.tolist()}"
    assert pv_col  in df.columns, f"P-value column '{pv_col}' not found. Got: {df.columns.tolist()}"

    # --- optional seed filtering + get a nice display label for the title ---
    display_seed_label = None
    if seed_col is not None and FILTER_BY_SEED is not None:
        assert seed_col in df.columns, f"Seed column '{seed_col}' not in CSV."
        seed_norm = norm_name(FILTER_BY_SEED)
        df["_seed_norm"] = df[seed_col].map(norm_name)
        # capture original seed label (first match) for title
        seed_series = df.loc[df["_seed_norm"] == seed_norm, seed_col]
        display_seed_label = str(seed_series.iloc[0]) if not seed_series.empty else FILTER_BY_SEED
        before = len(df)
        df = df[df["_seed_norm"] == seed_norm].copy()
        after = len(df)
        print(f"Filtered by seed == {FILTER_BY_SEED} → kept {after}/{before} rows.")
    # -----------------------------------------------------------------------

    # clean + dedupe: smallest p per ROI, keep ORIGINAL ROI name for hover
    df[pv_col] = pd.to_numeric(df[pv_col], errors="coerce")
    df = df.dropna(subset=[pv_col]).copy()
    df["_roi_norm"] = df[roi_col].map(norm_name)

    df_sorted = df.sort_values(["_roi_norm", pv_col], ascending=[True, True])
    # normalized ROI -> ORIGINAL CSV ROI name (for hover)
    display_map = df_sorted.drop_duplicates("_roi_norm").set_index("_roi_norm")[roi_col].to_dict()
    # min p per normalized ROI
    g = df_sorted.groupby("_roi_norm", as_index=False)[pv_col].min()

    # threshold (optional)
    if P_THRESH is not None:
        g = g[g[pv_col] < P_THRESH]

    print("Mapping CSV p-values to AAL indices...")
    roi_val_to_p = {}
    roi_val_to_intensity = {}
    roi_val_to_display = {}
    miss = []

    for _, r in g.iterrows():
        roi_norm = r["_roi_norm"]
        p = float(r[pv_col])

        val = name2val.get(roi_norm)
        if val is None:
            # try spaced variant
            val = name2val.get(norm_name(roi_norm.replace("_L"," Left").replace("_R"," Right")))
        if val is None:
            miss.append(roi_norm); continue

        roi_val_to_p[val]         = p
        roi_val_to_intensity[val] = max(-np.log10(p + EPS), 0.0)
        roi_val_to_display[val]   = display_map.get(roi_norm, roi_norm)  # exact CSV ROI for hover

    if miss:
        print("Unmatched ROI names (first 10):", miss[:10])
    if not roi_val_to_intensity:
        raise RuntimeError("No ROIs passed threshold/match. Set P_THRESH=None or check column names.")

    print("Fetching fsaverage surfaces (pial + white)...")
    fsavg = datasets.fetch_surf_fsaverage()
    pial_L, pial_R   = fsavg["pial_left"],  fsavg["pial_right"]
    white_L, white_R = fsavg["white_left"], fsavg["white_right"]

    # robust label sampling (white→pial line, nearest)
    def sample_labels(white_mesh, pial_mesh):
        try:
            lbl = surface.vol_to_surf(
                atlas_img, pial_mesh,
                inner_mesh=white_mesh, kind="line", n_samples=25, interpolation="nearest"
            )
        except TypeError:
            lbl = surface.vol_to_surf(atlas_img, pial_mesh)
        lbl = np.rint(np.asarray(lbl)).astype(np.int32)
        lbl[lbl < 0] = 0
        return lbl

    print("Sampling labels onto surface vertices...")
    lbl_L = sample_labels(white_L, pial_L)
    lbl_R = sample_labels(white_R, pial_R)
    cov_L = (lbl_L != 0).mean()*100; cov_R = (lbl_R != 0).mean()*100
    print(f"Nonzero label coverage: L {cov_L:.1f}% | R {cov_R:.1f}%")

    # per-vertex data from ROI dictionaries
    def build_vertex_data(surf_mesh, labels_on_vertices):
        coords, faces = surface.load_surf_mesh(surf_mesh)
        intens = np.zeros(coords.shape[0], dtype=float)
        pvals  = np.full(coords.shape[0], np.nan, dtype=float)
        names  = np.empty(coords.shape[0], dtype=object)

        for val, inten in roi_val_to_intensity.items():
            mask = (labels_on_vertices == val)
            if not np.any(mask):
                continue
            intens[mask] = inten
            pvals[mask]  = roi_val_to_p[val]
            names[mask]  = roi_val_to_display.get(val, "")

        return coords, faces, intens, pvals, names

    coords_L, faces_L, intens_L, p_L, names_L = build_vertex_data(pial_L, lbl_L)
    coords_R, faces_R, intens_R, p_R, names_R = build_vertex_data(pial_R, lbl_R)

    # shift hemispheres for a clean separation
    coords_L_shift = coords_L.copy(); coords_L_shift[:,0] -= HEMISPHERE_SHIFT
    coords_R_shift = coords_R.copy(); coords_R_shift[:,0] += HEMISPHERE_SHIFT

    # color scale range from non-zero intensities
    all_intens = np.concatenate([
        intens_L[~np.isnan(intens_L)], intens_R[~np.isnan(intens_R)]
    ])
    if all_intens.size == 0 or np.nanmax(all_intens) == 0:
        raise RuntimeError("All intensities are zero/NaN after filtering; relax P_THRESH or adjust inputs.")
    cmin, cmax = float(np.nanmin(all_intens)), float(np.nanmax(all_intens))

    # background gray surfaces (no hover), then colored ROIs on top
    lighting = dict(ambient=0.35, diffuse=0.6, specular=0.2, roughness=0.8, fresnel=0.2)
    fig = go.Figure()

    # background layers
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:,0], y=coords_L_shift[:,1], z=coords_L_shift[:,2],
        i=faces_L[:,0], j=faces_L[:,1], k=faces_L[:,2],
        color="lightgray", opacity=0.25, name="background L",
        hoverinfo="skip", showscale=False, lighting=dict(ambient=0.5, diffuse=0.5)
    ))
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:,0], y=coords_R_shift[:,1], z=coords_R_shift[:,2],
        i=faces_R[:,0], j=faces_R[:,1], k=faces_R[:,2],
        color="lightgray", opacity=0.25, name="background R",
        hoverinfo="skip", showscale=False, lighting=dict(ambient=0.5, diffuse=0.5)
    ))

    # build customdata arrays for hover
    if SHOW_P_IN_HOVER:
        custom_L = np.stack([
            np.where(names_L == None, "", names_L.astype(str)),
            np.where(np.isnan(p_L), np.nan, p_L)
        ], axis=1)
        custom_R = np.stack([
            np.where(names_R == None, "", names_R.astype(str)),
            np.where(np.isnan(p_R), np.nan, p_R)
        ], axis=1)
        hover_tmpl = "<b>%{customdata[0]}</b><br>p = %{customdata[1]:.3g}<extra></extra>"
    else:
        custom_L = np.where(names_L == None, "", names_L.astype(str)).reshape(-1, 1)
        custom_R = np.where(names_R == None, "", names_R.astype(str)).reshape(-1, 1)
        hover_tmpl = "<b>%{customdata[0]}</b><extra></extra>"

    # colored ROI layers
    fig.add_trace(go.Mesh3d(
        x=coords_L_shift[:,0], y=coords_L_shift[:,1], z=coords_L_shift[:,2],
        i=faces_L[:,0], j=faces_L[:,1], k=faces_L[:,2],
        intensity=intens_L, cmin=cmin, cmax=cmax, colorscale=COLORSCALE,
        opacity=OPACITY, flatshading=False, lighting=lighting,
        showscale=True, name="Left hemisphere",
        hovertemplate=hover_tmpl, customdata=custom_L,
        colorbar=dict(
            title="-log10(p)", len=0.80,
            tickvals=[-np.log10(x) for x in (0.05, 0.01, 0.001)],
            ticktext=["0.05","0.01","0.001"]
        )
    ))
    fig.add_trace(go.Mesh3d(
        x=coords_R_shift[:,0], y=coords_R_shift[:,1], z=coords_R_shift[:,2],
        i=faces_R[:,0], j=faces_R[:,1], k=faces_R[:,2],
        intensity=intens_R, cmin=cmin, cmax=cmax, colorscale=COLORSCALE,
        opacity=OPACITY, flatshading=False, lighting=lighting,
        showscale=False, name="Right hemisphere",
        hovertemplate=hover_tmpl, customdata=custom_R
    ))

    # dynamic title includes seed (if set) and threshold
    title_parts = ["fsaverage cortex – AAL ROI p-map"]
    if seed_col is not None and FILTER_BY_SEED is not None:
        title_parts.append(f"seed: {display_seed_label or FILTER_BY_SEED}")
    if P_THRESH is not None:
        title_parts.append(f"p < {P_THRESH:g}")
    fig_title = " | ".join(title_parts)

    fig.update_layout(
        title=fig_title,
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7))
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.02)
    )

    # offline, self-contained HTML
    fig.write_html(OUT_HTML, include_plotlyjs="inline", full_html=True, auto_open=False)
    print(f"Wrote: {OUT_HTML}")

    if SAVE_PNG:
        try:
            fig.write_image(PNG_PATH, width=2000, height=1400, scale=2)
            print(f"Wrote: {PNG_PATH}")
        except Exception as e:
            print("PNG export needs 'kaleido' (pip install -U kaleido). Error:", e)

if __name__ == "__main__":
    main()
