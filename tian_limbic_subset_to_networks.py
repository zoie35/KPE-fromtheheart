#!/usr/bin/env python3
"""
Subcortical limbic seeds  ->  Schaefer Control & Default(DMN) networks
=====================================================================

Reproduces the region-to-network design from Siegel et al. 2024 (Nature,
"Psilocybin desynchronizes the human brain"), which defined a subcortical
limbic network from anatomy and compared a seed (anterior hippocampus) to a
whole cortical network (DMN).

Here we:
  1. Group the Tian S2 subcortical ROIs into 5 named limbic seeds
     (amygdala, anterior hippocampus, posterior hippocampus,
      anteromedial thalamus, nucleus accumbens).
  2. "Correlate then average": for each seed, correlate EVERY member ROI with
     EVERY parcel of the target network, Fisher-z each correlation, then
     average all of them -> one "seed -> network" FC value (mean z) per scan.
     (We deliberately do NOT average the member ROI timeseries together first,
     because that would wash out signal from sub-regions such as lateral vs
     medial amygdala or shell vs core accumbens.)
  3. Target networks are the Schaefer Control network AND the Schaefer
     Default/DMN network.
  4. Compute baseline(ses-1) -> follow-up delta per subject.
  5. Test ketamine vs placebo on the deltas (nominal p + FDR).

This is ADDITIVE and SAFE:
  - It writes NEW, timestamped files into tian_control_results/.
  - It does NOT touch or overwrite any existing result files.

It reuses the already-working helpers from tian_to_schaefer_limbic_control.py
(randomization loading, file matching, timeseries loading, stats) so behaviour
stays consistent with your existing pipeline.

Run it from the Mac Terminal with:
    cd /Users/zoiemilstein/projects/KPE-fromtheheart
    .venv/bin/python tian_limbic_subset_to_networks.py
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt

# Reuse the proven helpers from the existing Tian script.
import tian_to_schaefer_limbic_control as tsc


# ============================================================
# USER SETTINGS
# ============================================================

# Same dataset roots as the existing Tian script.
DATASETS = tsc.DATASETS
PIPELINES = tsc.PIPELINES          # ["global", "anatomical"]
FOLLOWUPS = tsc.FOLLOWUPS          # ["ses-2", "ses-3"]
OUTPUT_DIR = tsc.OUTPUT_DIR        # kpe/tian_control_results

# The 5 subcortical limbic seeds, mapped to Tian S2 column labels.
# Left + right are averaged together into one bilateral seed.
LIMBIC_SEEDS = {
    "amygdala":              ["lAMY-lh", "mAMY-lh", "lAMY-rh", "mAMY-rh"],
    "ant_hippocampus":       ["aHIP-lh", "aHIP-rh"],
    "post_hippocampus":      ["pHIP-lh", "pHIP-rh"],
    "anteromedial_thalamus": ["THA-DA-lh", "THA-VA-lh", "THA-DA-rh", "THA-VA-rh"],
    "nucleus_accumbens":     ["NAc-shell-lh", "NAc-core-lh", "NAc-shell-rh", "NAc-core-rh"],
}

# Target Schaefer 7-networks, matched by substring in the column name.
# Schaefer labels look like "7Networks_LH_Cont_..." and "7Networks_LH_Default_...".
TARGET_NETWORKS = {
    "Control": "_Cont_",
    "DMN":     "_Default_",
}

MIN_TIMEPOINTS = 10  # skip a scan/seed if fewer usable timepoints than this


# ============================================================
# CORE COMPUTATION
# ============================================================

def build_seed_members(tian_df: pd.DataFrame):
    """
    For each limbic seed, list the member Tian columns that are actually present.
    We keep the members SEPARATE (we do not average their timeseries) so that
    each sub-region's signal is preserved until the final z-averaging step.

    Returns:
        seeds_present: dict {seed_name: [member column names present in this file]}
        missing_report: dict {seed_name: [columns that were expected but missing]}
    """
    seeds_present = {}
    missing_report = {}

    available = set(tian_df.columns)

    for seed_name, columns in LIMBIC_SEEDS.items():
        present = [c for c in columns if c in available]
        missing = [c for c in columns if c not in available]

        missing_report[seed_name] = missing

        if len(present) == 0:
            continue

        seeds_present[seed_name] = present

    return seeds_present, missing_report


def get_network_columns(schaefer_df: pd.DataFrame):
    """Return {network_name: [schaefer columns in that network]}."""
    out = {}
    for net_name, tag in TARGET_NETWORKS.items():
        cols = [c for c in schaefer_df.columns if tag in str(c)]
        out[net_name] = cols
    return out


def seed_to_network_mean_z(member_df: pd.DataFrame, network_df: pd.DataFrame):
    """
    "Correlate then average."

    Correlate EVERY seed member ROI with EVERY network parcel (within this one
    scan), Fisher-z each correlation, then return the mean z across all
    member x parcel pairs. This keeps each sub-region's signal intact and only
    pools them at the very end.

    Returns:
        mean_z, n_edges  (n_edges = number of member x parcel correlations used)
    """
    if member_df.shape[1] == 0 or network_df.shape[1] == 0:
        return np.nan, 0

    members = member_df.to_numpy(dtype=float)   # T x M
    net = network_df.to_numpy(dtype=float)      # T x P

    # Align lengths (same scan, but guard anyway).
    n = min(members.shape[0], net.shape[0])
    members = members[:n, :]
    net = net[:n, :]

    # Drop timepoints with any NaN in either side.
    valid_rows = np.isfinite(members).all(axis=1) & np.isfinite(net).all(axis=1)
    members = members[valid_rows, :]
    net = net[valid_rows, :]

    if members.shape[0] < MIN_TIMEPOINTS:
        return np.nan, 0

    # Standardize each column; correlation = normalized dot product.
    m_sd = members.std(axis=0, ddof=1)
    p_sd = net.std(axis=0, ddof=1)

    m_ok = m_sd > 0
    p_ok = p_sd > 0
    if not m_ok.any() or not p_ok.any():
        return np.nan, 0

    M = (members[:, m_ok] - members[:, m_ok].mean(axis=0)) / m_sd[m_ok]
    P = (net[:, p_ok] - net[:, p_ok].mean(axis=0)) / p_sd[p_ok]

    # M: T x m, P: T x p  ->  corr matrix m x p
    corr = np.dot(M.T, P) / (M.shape[0] - 1)
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)

    return float(np.mean(z)), int(z.size)


# ============================================================
# STEP 1: SCAN-LEVEL TABLE
# ============================================================

def build_scan_level_table():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: Building scan-level limbic-seed -> network table")
    print("=" * 80)
    print(f"Datasets : {list(DATASETS.keys())}")
    print(f"Pipelines: {PIPELINES}")
    print(f"Seeds    : {list(LIMBIC_SEEDS.keys())}")
    print(f"Networks : {list(TARGET_NETWORKS.keys())}")
    print()

    # Load randomization once per dataset (reuse the existing loader).
    randomization_maps = {
        dataset: tsc.read_randomization_map(root, dataset)
        for dataset, root in DATASETS.items()
    }

    records = []
    problems = []
    n_scans_seen = 0
    missing_columns_warned = set()

    for dataset, root in DATASETS.items():
        rand_map = randomization_maps[dataset]

        for pipeline in PIPELINES:
            folder = root / pipeline

            if not folder.exists():
                print(f"WARNING: folder does not exist: {folder}")
                continue

            tian_files = sorted(folder.glob("*tian_s2_ts.csv"))
            print(f"[{dataset} | {pipeline}] Tian files found: {len(tian_files)}")

            for tian_file in tian_files:
                n_scans_seen += 1
                info = tsc.parse_filename(tian_file)
                subject = info["subject"]
                session = info["session"]

                dose_group = tsc.get_group_for_subject(subject, rand_map)
                bin_group = tsc.binary_group(dose_group)

                if dose_group is None:
                    problems.append({"dataset": dataset, "pipeline": pipeline,
                                     "subject": subject, "session": session,
                                     "file": tian_file.name,
                                     "problem": "No randomization group found"})
                    continue

                schaefer_file = tsc.find_matching_schaefer_file(tian_file)
                if schaefer_file is None:
                    problems.append({"dataset": dataset, "pipeline": pipeline,
                                     "subject": subject, "session": session,
                                     "file": tian_file.name,
                                     "problem": "No matching Schaefer file found"})
                    continue

                try:
                    tian_ts = tsc.load_timeseries_csv(tian_file)
                    schaefer_ts = tsc.load_timeseries_csv(schaefer_file)

                    # Skip clearly-broken Schaefer files (expect 400 parcels).
                    if schaefer_ts.shape[1] < 300:
                        problems.append({"dataset": dataset, "pipeline": pipeline,
                                         "subject": subject, "session": session,
                                         "file": schaefer_file.name,
                                         "problem": f"Schaefer has too few columns: {schaefer_ts.shape[1]}"})
                        continue

                    seeds, missing_report = build_seed_members(tian_ts)
                    net_cols = get_network_columns(schaefer_ts)

                    # Warn once about any seed missing all its columns.
                    for seed_name, missing in missing_report.items():
                        if missing and seed_name not in seeds:
                            key = (dataset, pipeline, seed_name)
                            if key not in missing_columns_warned:
                                print(f"  NOTE [{dataset}|{pipeline}] seed '{seed_name}' "
                                      f"missing columns {missing}")
                                missing_columns_warned.add(key)

                    for seed_name, member_cols in seeds.items():
                        for net_name, cols in net_cols.items():
                            mean_z, n_edges = seed_to_network_mean_z(
                                tian_ts[member_cols], schaefer_ts[cols]
                            )
                            records.append({
                                "dataset": dataset,
                                "pipeline": pipeline,
                                "subject": subject,
                                "session": session,
                                "session_raw": info["session_raw"],
                                "run": info["run"],
                                "dose_group": dose_group,
                                "binary_group": bin_group,
                                "seed": seed_name,
                                "network": net_name,
                                "seed_to_network_z": mean_z,
                                "n_edges": n_edges,
                                "tian_file": tian_file.name,
                                "schaefer_file": schaefer_file.name,
                            })

                except Exception as e:
                    problems.append({"dataset": dataset, "pipeline": pipeline,
                                     "subject": subject, "session": session,
                                     "file": tian_file.name, "problem": repr(e)})

    scan_df = pd.DataFrame(records)
    problems_df = pd.DataFrame(problems)

    print()
    print(f"Scans seen        : {n_scans_seen}")
    print(f"Scan x seed x net rows: {len(scan_df)}")
    print(f"Problem scans     : {len(problems_df)}")
    if not problems_df.empty:
        print("Problem summary:")
        print(problems_df["problem"].value_counts().to_string())
    print()

    return scan_df, problems_df


# ============================================================
# STEP 2: DELTA TABLE (baseline -> follow-up)
# ============================================================

def compute_delta_table(scan_df: pd.DataFrame):
    if scan_df.empty:
        return pd.DataFrame()

    # Average repeated runs within a subject/session/seed/network.
    avg = (scan_df
           .groupby(["dataset", "pipeline", "subject", "session",
                     "dose_group", "binary_group", "seed", "network"],
                    as_index=False)["seed_to_network_z"]
           .mean())

    baseline = avg[avg["session"] == "ses-1"].copy()
    baseline = baseline.rename(columns={"seed_to_network_z": "z_baseline"}).drop(columns=["session"])

    all_deltas = []
    for followup in FOLLOWUPS:
        follow = avg[avg["session"] == followup].copy()
        follow = follow.rename(columns={"seed_to_network_z": "z_followup"}).drop(columns=["session"])

        merged = follow.merge(
            baseline,
            on=["dataset", "pipeline", "subject", "dose_group", "binary_group", "seed", "network"],
            how="inner",
        )
        merged["session_contrast"] = f"ses-1_to_{followup}"
        merged["delta_z"] = merged["z_followup"] - merged["z_baseline"]
        all_deltas.append(merged)

    delta_df = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()
    print(f"STEP 2: delta rows (subjects with baseline + follow-up): {len(delta_df)}")
    print()
    return delta_df


# ============================================================
# STEP 3: KETAMINE vs PLACEBO STATS
# ============================================================

def run_group_statistics(delta_df: pd.DataFrame):
    if delta_df.empty:
        print("STEP 3: no deltas -> no stats.")
        return pd.DataFrame()

    # All four ketamine/placebo contrasts, matching the original Tian script.
    COMPARISONS = [
        {"comparison": "ketamine_0.5+0.2_vs_placebo", "group_col": "binary_group",
         "group1": "ketamine",     "group2": "placebo"},
        {"comparison": "ketamine_0.5_vs_placebo",     "group_col": "dose_group",
         "group1": "ketamine_0.5", "group2": "placebo"},
        {"comparison": "ketamine_0.2_vs_placebo",     "group_col": "dose_group",
         "group1": "ketamine_0.2", "group2": "placebo"},
        {"comparison": "ketamine_0.5_vs_ketamine_0.2","group_col": "dose_group",
         "group1": "ketamine_0.5", "group2": "ketamine_0.2"},
    ]

    records = []
    scopes = ["combined"] + sorted(delta_df["dataset"].dropna().unique())

    for scope in scopes:
        scope_df = delta_df if scope == "combined" else delta_df[delta_df["dataset"] == scope]

        for comp in COMPARISONS:
            for (pipeline, network, seed, contrast), sub in scope_df.groupby(
                ["pipeline", "network", "seed", "session_contrast"]
            ):
                result = tsc.compare_two_groups(
                    sub, group_col=comp["group_col"],
                    group1=comp["group1"], group2=comp["group2"]
                )
                if result is None:
                    continue
                records.append({
                    "analysis_dataset": scope,
                    "pipeline": pipeline,
                    "network": network,
                    "seed": seed,
                    "session_contrast": contrast,
                    "comparison": comp["comparison"],
                    **result,
                })

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        print("STEP 3: no valid comparisons (need >=2 subjects per group).")
        return stats_df

    # FDR within each family (dataset x comparison x pipeline x network x contrast).
    stats_df["p_fdr"] = np.nan
    family_cols = ["analysis_dataset", "comparison", "pipeline", "network", "session_contrast"]
    for _, idx in stats_df.groupby(family_cols).groups.items():
        idx = list(idx)
        stats_df.loc[idx, "p_fdr"] = tsc.fdr_bh(stats_df.loc[idx, "p_uncorrected"].values)

    stats_df["significant_fdr_0.05"] = stats_df["p_fdr"] < 0.05
    stats_df = stats_df.sort_values("p_uncorrected")
    print(f"STEP 3: stats rows: {len(stats_df)}")
    print()
    return stats_df


# ============================================================
# STEP 4: VISUALIZATION
# ============================================================

def make_figure(scan_df: pd.DataFrame, stats_df: pd.DataFrame, fig_path: Path):
    """
    Two panels:
      A. Mean baseline (ses-1) seed->network FC by pipeline (the connectivity structure).
      B. Ketamine - placebo mean delta (combined, ses-1 -> ses-3) with p-values.
    """
    seeds = list(LIMBIC_SEEDS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel A: baseline connectivity structure ----
    base = scan_df[scan_df["session"] == "ses-1"]
    axA = axes[0]
    x = np.arange(len(seeds))
    width = 0.2
    combos = [("anatomical", "Control"), ("anatomical", "DMN"),
              ("global", "Control"), ("global", "DMN")]
    colors = ["#4C72B0", "#A6C0E4", "#C44E52", "#E7A6A8"]

    for i, (pipe, net) in enumerate(combos):
        means = []
        for seed in seeds:
            vals = base[(base["pipeline"] == pipe) & (base["network"] == net) &
                        (base["seed"] == seed)]["seed_to_network_z"].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
        axA.bar(x + (i - 1.5) * width, means, width,
                label=f"{pipe} - {net}", color=colors[i])

    axA.set_xticks(x)
    axA.set_xticklabels([s.replace("_", "\n") for s in seeds], fontsize=8)
    axA.set_ylabel("Mean seed -> network FC (Fisher z)")
    axA.set_title("A. Baseline limbic-seed to network connectivity")
    axA.axhline(0, color="black", lw=0.8)
    axA.legend(fontsize=8)

    # ---- Panel B: ketamine - placebo delta (combined, ses-1 -> ses-3) ----
    axB = axes[1]
    # Panel B shows the PRIMARY comparison only (combined ketamine 0.5+0.2 vs placebo).
    if not stats_df.empty:
        sub = stats_df[(stats_df["analysis_dataset"] == "combined") &
                       (stats_df["comparison"] == "ketamine_0.5+0.2_vs_placebo") &
                       (stats_df["session_contrast"] == "ses-1_to_ses-3")]
    else:
        sub = pd.DataFrame()

    if not sub.empty:
        labels, diffs, pvals, bar_colors = [], [], [], []
        for pipe in ["anatomical", "global"]:
            for net in ["Control", "DMN"]:
                for seed in seeds:
                    row = sub[(sub["pipeline"] == pipe) & (sub["network"] == net) &
                              (sub["seed"] == seed)]
                    if row.empty:
                        continue
                    labels.append(f"{seed}\n{pipe[:4]}-{net}")
                    diffs.append(row["difference_group1_minus_group2"].iloc[0])
                    pvals.append(row["p_uncorrected"].iloc[0])
                    bar_colors.append("#C44E52" if row["p_uncorrected"].iloc[0] < 0.05 else "#BBBBBB")

        y = np.arange(len(labels))
        axB.barh(y, diffs, color=bar_colors)
        axB.set_yticks(y)
        axB.set_yticklabels(labels, fontsize=6)
        axB.axvline(0, color="black", lw=0.8)
        axB.set_xlabel("Ketamine - placebo mean delta (Fisher z)")
        axB.set_title("B. Ketamine vs placebo change\n(combined, ses-1 -> ses-3)")
        for i, p in enumerate(pvals):
            axB.text(diffs[i], i, f" p={p:.2f}", va="center", fontsize=5)
    else:
        axB.text(0.5, 0.5, "No group stats available", ha="center", va="center")
        axB.set_title("B. Ketamine vs placebo change")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scan_df, problems_df = build_scan_level_table()
    if scan_df.empty:
        print("No scan-level rows produced. Check the problems table. Stopping.")
        return

    delta_df = compute_delta_table(scan_df)
    stats_df = run_group_statistics(delta_df)

    # Save NEW timestamped files (nothing overwritten).
    scan_path = OUTPUT_DIR / f"scan_level_limbic_subset_to_networks_{stamp}.csv"
    delta_path = OUTPUT_DIR / f"delta_limbic_subset_to_networks_{stamp}.csv"
    stats_path = OUTPUT_DIR / f"stats_limbic_subset_to_networks_{stamp}.csv"
    fig_path = OUTPUT_DIR / f"figure_limbic_subset_to_networks_{stamp}.png"

    scan_df.to_csv(scan_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    if not stats_df.empty:
        stats_df.to_csv(stats_path, index=False)
    if not problems_df.empty:
        problems_df.to_csv(
            OUTPUT_DIR / f"problems_limbic_subset_to_networks_{stamp}.csv", index=False
        )

    make_figure(scan_df, stats_df, fig_path)

    print()
    print("=" * 80)
    print("DONE. Files written:")
    print(f"  {scan_path.name}")
    print(f"  {delta_path.name}")
    if not stats_df.empty:
        print(f"  {stats_path.name}")
    print(f"  {fig_path.name}")
    print(f"All in: {OUTPUT_DIR}")
    print("=" * 80)

    if not stats_df.empty:
        print("\nTop 10 nominal results:")
        cols = ["analysis_dataset", "pipeline", "network", "seed", "session_contrast",
                "n_group1", "n_group2", "difference_group1_minus_group2",
                "p_uncorrected", "p_fdr"]
        print(stats_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
