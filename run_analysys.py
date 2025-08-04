import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib
from nilearn.plotting import plot_connectome, plot_matrix, show
# Settings

#project_root = r"C:\kpeSoundPath\KPE-fromtheheart"  # Use the correct path
project_root = r"C:\kpeSoundPath\KPE-fromtheheart"  # Use the correct path

output_folder = os.path.join(project_root, "t_test_results")
os.makedirs(output_folder, exist_ok=True)
# --- NEW: group comparison (add near your imports) ---
from scipy.stats import ttest_ind
import re
show_correlation_matrix = False

def load_group_table(xlsx_path, subject_col="SubID", group_col="Group_Simbol",
                     ketamine_groups=("A","B"), control_groups=("C",)):
    df = pd.read_excel(xlsx_path)
    # Normalize subject ids to match filenames like 'sub-024'
    def norm_sub(x):
        x = str(x).strip()
        if re.match(r"sub-\d+", x):
            return x
        # try to coerce numbers to sub-XYZ with zero pad to 3
        m = re.match(r".*?(\d+)", x)
        return f"sub-{int(m.group(1)):03d}" if m else x

    df["_sub_norm"] = df[subject_col].apply(norm_sub)
    df["_group_norm"] = df[group_col].astype(str).str.strip().str.upper()
    group_map = {}
    for _, row in df.iterrows():
        g = row["_group_norm"]
        if g in ketamine_groups:
            group_map[row["_sub_norm"]] = "ketamine"
        elif g in control_groups:
            group_map[row["_sub_norm"]] = "control"
        else:
            group_map[row["_sub_norm"]] = None
    return group_map, df

def compute_subject_deltas(amygdala_correlations):
    """
    Returns nested dict:
      deltas[(seed, region)][subject] = delta (MRI3 - MRI1)
    Requires subjects to have both MRI1 and MRI3.
    """
    # reorganize by session
    sess = {}
    for (sub, session), corr in amygdala_correlations.items():
        sess.setdefault(session, {})[sub] = corr

    # identify MRI1 and MRI3
    s1 = next((s for s in sess if "MRI1" in s or "S1" in s.upper()), None)
    s3 = next((s for s in sess if "MRI3" in s or "S3" in s.upper()), None)
    if not s1 or not s3:
        raise ValueError("Could not find both MRI1 and MRI3 sessions.")

    common = set(sess[s1].keys()) & set(sess[s3].keys())
    if not common:
        raise ValueError("No subjects with both MRI1 and MRI3.")

    # collect all regions from MRI1 (Amygdala_L keys)
    all_regions = set()
    for sub in common:
        all_regions.update(sess[s1][sub]['Amygdala_L'].index)

    deltas = {}
    for seed in ['Amygdala_L', 'Amygdala_R']:
        for region in all_regions:
            key = (seed, region)
            deltas[key] = {}
            for sub in common:
                if region in sess[s1][sub][seed] and region in sess[s3][sub][seed]:
                    d = float(sess[s3][sub][seed][region]) - float(sess[s1][sub][seed][region])
                    deltas[key][sub] = d
    return deltas

def cohen_d_independent(x, y):
    # Cohen's d (pooled SD); handle small n
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)) if (nx+ny-2) > 0 else np.nan
    return (np.mean(x) - np.mean(y)) / sp if sp and not np.isnan(sp) and sp != 0 else np.nan

def between_group_tests(amygdala_correlations, randomization_xlsx, output_folder,
                        subject_col="SubID", group_col="Group_Simbol"):
    group_map, _ = load_group_table(randomization_xlsx, subject_col, group_col)
    deltas = compute_subject_deltas(amygdala_correlations)

    rows = []
    for (seed, region), sub_to_delta in deltas.items():
        # partition by group
        ket = []; ctrl = []
        for sub, val in sub_to_delta.items():
            grp = group_map.get(sub)
            if grp == "ketamine":
                ket.append(val)
            elif grp == "control":
                ctrl.append(val)
        if len(ket) >= 2 and len(ctrl) >= 2:
            t, p = ttest_ind(ket, ctrl, equal_var=False)  # Welch's t-test
            d = cohen_d_independent(ket, ctrl)
            rows.append({
                "seed": seed,
                "region": region,
                "n_ketamine": len(ket),
                "n_control": len(ctrl),
                "mean_delta_ketamine": np.mean(ket),
                "mean_delta_control": np.mean(ctrl),
                "mean_diff_(ket-control)": np.mean(ket) - np.mean(ctrl),
                "t_statistic": t,
                "p_value": p,
                "cohens_d": d
            })

    out = pd.DataFrame(rows).sort_values("p_value")
    if not out.empty:
        out_path = os.path.join(output_folder, "between_group_MRI3_minus_MRI1_amygdala.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved between-group results to: {out_path}")
    else:
        print("No regions had sufficient subjects in both groups for testing.")
    return out

# --- (Optional) very simple volcano plot (one per seed) ---
def volcano_plot(results_df, output_folder):
    if results_df is None or results_df.empty:
        return
    for seed in results_df["seed"].unique():
        df = results_df[results_df["seed"] == seed].copy()
        if df.empty:
            continue
        df["neglog10p"] = -np.log10(df["p_value"])
        plt.figure(figsize=(8,6))
        plt.scatter(df["mean_diff_(ket-control)"], df["neglog10p"])
        plt.axhline(-np.log10(0.05), linestyle="--")
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("Mean Δ difference (ketamine − control)")
        plt.ylabel("−log10 p")
        plt.title(f"{seed}: MRI3−MRI1 group difference")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"volcano_{seed}.png"), dpi=300, bbox_inches="tight")
        plt.show()
def PearsonCorr(project_root):
    """Create correlation matrices from time series files"""
    correlation_matrix = {}

    # Look in the project_root directory for the actual data files
    for ts_file in os.listdir(project_root):
        if ts_file.endswith('.csv'):
            file_path = os.path.join(project_root, ts_file)
            try:
                df = pd.read_csv(file_path)
                # Verify we have the expected structure
                if 'Amygdala_L' in df.columns and 'Amygdala_R' in df.columns:
                    corr = df.corr()
                    correlation_matrix[ts_file] = corr
                    print(f"Created correlation matrix for {ts_file}")
                else:
                    print(f"Warning: {ts_file} missing amygdala columns")
            except Exception as e:
                print(f"Error processing {ts_file}: {e}")
    if (show_correlation_matrix):
        for filename, corrMat in correlation_matrix.items():
            plot_matrix(corrMat, vmax=0.8, vmin=-0.8, colorbar=True)
            plt.title(filename, fontsize=10)
            plt.tight_layout()
            plt.show()
    return correlation_matrix


def export_correlation_matrices(correlation_matrix, output_folder):

    """Export correlation matrices to CSV files"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\n=== Exporting Correlation Matrices ===")

    for file_name, corr_df in correlation_matrix.items():
        # Create output filename
        base_name = file_name.replace('_aal_ts.csv', '_correlation_matrix.csv')
        output_path = os.path.join(output_folder, base_name)

        # Save correlation matrix
        corr_df.to_csv(output_path, index=True)
        print(f"Exported: {base_name}")

    print(f"All correlation matrices saved to: {output_folder}")


def extract_amygdala_correlations(correlation_matrix):
    """Extract left and right amygdala correlations"""
    amygdala_correlations = {}

    for file_name, corr_df in correlation_matrix.items():
        left_amygdala_corr = corr_df.loc['Amygdala_L', :].drop('Amygdala_L')
        right_amygdala_corr = corr_df.loc['Amygdala_R', :].drop('Amygdala_R')

        parts = file_name.replace("_aal_ts.csv", "").split("_")
        subject = parts[0]
        session = parts[1]  # This will be ses-MRI1, ses-MRI2, ses-MRI3

        amygdala_correlations[(subject, session)] = {
            'Amygdala_L': left_amygdala_corr,
            'Amygdala_R': right_amygdala_corr
        }

    return amygdala_correlations


def analyze_session_changes(amygdala_correlations):
    """Compare MRI1 vs MRI3 for both amygdalae"""
    # Organize by session
    session_data = {}
    for (subject, session), correlations in amygdala_correlations.items():
        if session not in session_data:
            session_data[session] = []
        session_data[session].append({
            'subject': subject,
            'Amygdala_L': correlations['Amygdala_L'],
            'Amygdala_R': correlations['Amygdala_R']
        })

    # Check what sessions are available
    available_sessions = list(session_data.keys())
    print(f"Available sessions: {available_sessions}")

    # Use MRI1 vs MRI3 sessions
    if len(available_sessions) < 2:
        print("Need at least 2 sessions for comparison")
        return None

    # Look for MRI1 and MRI3 sessions
    mri1_session = None
    mri3_session = None

    for session in available_sessions:
        if 'MRI1' in session:
            mri1_session = session
        elif 'MRI3' in session:
            mri3_session = session

    if not mri1_session or not mri3_session:
        print("Need both MRI1 and MRI3 sessions")
        return None

    print(f"Comparing {mri1_session} vs {mri3_session}")

    mri1_data = session_data.get(mri1_session, [])
    mri3_data = session_data.get(mri3_session, [])

    if not mri1_data or not mri3_data:
        print(f"Need both {mri1_session} and {mri3_session} data")
        return None

    # Get subjects with both sessions
    mri1_subjects = {data['subject'] for data in mri1_data}
    mri3_subjects = {data['subject'] for data in mri3_data}
    common_subjects = mri1_subjects.intersection(mri3_subjects)

    print(f"Subjects with both MRI1 and MRI3: {len(common_subjects)}")

    if len(common_subjects) < 1:
        print("Need at least 1 subject for comparison")
        return None

    # Get all regions
    all_regions = set()
    for data in mri1_data + mri3_data:
        all_regions.update(data['Amygdala_L'].keys())

    print(f"Total unique regions found: {len(all_regions)}")
    print(f"Sample regions: {list(all_regions)[:5]}")

    # Compare each region
    results = []

    for region in all_regions:
        # Analyze both amygdalae
        for seed_type in ['Amygdala_L', 'Amygdala_R']:
            mri1_values = []
            mri3_values = []

            for subject in common_subjects:
                mri1_subj_data = next(data for data in mri1_data if data['subject'] == subject)
                if region in mri1_subj_data[seed_type]:
                    mri1_values.append(mri1_subj_data[seed_type][region])

                mri3_subj_data = next(data for data in mri3_data if data['subject'] == subject)
                if region in mri3_subj_data[seed_type]:
                    mri3_values.append(mri3_subj_data[seed_type][region])

            # Analyze if enough data
            if len(mri1_values) >= 1 and len(mri3_values) >= 1:
                seed_name = 'Amygdala_L' if seed_type == 'Amygdala_L' else 'Amygdala_R'
                print(f"Analyzing {region} for {seed_name}: {len(mri1_values)} subjects")
                t_stat, p_val = ttest_rel(mri1_values, mri3_values)
                mean_diff = np.mean(mri3_values) - np.mean(mri1_values)
                print(f"Mean difference: {mean_diff}")
                print(f"Region: {region}")
                results.append({
                    'region': region,
                    'seed': seed_name,
                    'mri1_mean': np.mean(mri1_values),
                    'mri3_mean': np.mean(mri3_values),
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'n_subjects': len(mri1_values)
                })

    return pd.DataFrame(results)


def plot_session_changes(results_df, output_folder):
    """Create graphs for left and right amygdala session changes"""
    if results_df is None or len(results_df) == 0:
        print("No results to plot")
        return

    # Filter for more significant changes (p < 0.05) and larger effect sizes
    significant_results = results_df[
        (results_df['p_value'] < 0.05) &
        (results_df['t_statistic'].abs() > 2.0)  # Only show substantial changes
        ].copy()

    if len(significant_results) == 0:
        print("No significant changes found (p < 0.05, |t| > 2.0), showing top 10 by t-statistic")
        significant_results = results_df.nlargest(10, 't_statistic').copy()

    # Create plots for each amygdala
    for seed in ['Amygdala_L', 'Amygdala_R']:
        seed_results = significant_results[significant_results['seed'] == seed].copy()

        if len(seed_results) == 0:
            print(f"No data for {seed}")
            continue

        # Sort by absolute t-statistic
        seed_results['abs_t_stat'] = seed_results['t_statistic'].abs()
        seed_results = seed_results.sort_values('abs_t_stat', ascending=False)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: T-statistics
        colors = ['red' if row['t_statistic'] > 0 else 'blue' for _, row in seed_results.iterrows()]
        ax1.barh(range(len(seed_results)), seed_results['t_statistic'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(seed_results)))
        ax1.set_yticklabels([region[:25] + '...' if len(region) > 25 else region for region in seed_results['region']])
        ax1.set_xlabel('T-Statistic')
        ax1.set_title(f'{seed} - T-Statistics (MRI3 vs MRI1)')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Add p-value annotations
        for i, (_, row) in enumerate(seed_results.iterrows()):
            p_text = f"p={row['p_value']:.3f}"
            ax1.text(row['t_statistic'], i, f' {p_text}', va='center', fontsize=8)

        # Plot 2: P-values
        ax2.barh(range(len(seed_results)), -np.log10(seed_results['p_value']), color=colors, alpha=0.7)
        ax2.set_yticks(range(len(seed_results)))
        ax2.set_yticklabels([region[:25] + '...' if len(region) > 25 else region for region in seed_results['region']])
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title(f'{seed} - Statistical Significance')
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax2.axvline(x=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{seed.replace(" ", "_")}_session_changes.png'), dpi=300,
                    bbox_inches='tight')
        plt.show()

        # Print summary
        print(f"\n=== {seed} Session Changes ===")
        print(f"Top 10 regions by absolute t-statistic:")
        for _, row in seed_results.head(10).iterrows():
            direction = "increased" if row['mean_difference'] > 0 else "decreased"
            print(f"  {row['region']}: t={row['t_statistic']:.3f}, p={row['p_value']:.4f}, {direction}")

    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    for seed in ['Amygdala_L', 'Amygdala_R']:
        seed_results = results_df[results_df['seed'] == seed]
        print(f"\n{seed}:")
        print(f"  Total regions tested: {len(seed_results)}")
        print(f"  Significant changes (p < 0.05): {len(seed_results[seed_results['p_value'] < 0.05])}")
        print(f"  Large changes (|t| > 2.0): {len(seed_results[seed_results['t_statistic'].abs() > 2.0])}")
        print(f"  Mean t-statistic: {seed_results['t_statistic'].mean():.3f}")
        print(f"  Mean p-value: {seed_results['p_value'].mean():.3f}")

    # Save results
    results_csv = os.path.join(output_folder, "session_changes_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved detailed results to: {results_csv}")


def create_brain_visualization(results_df, output_folder):
    """Create brain visualizations of the connectivity changes"""
    if results_df is None or len(results_df) == 0:
        print("No results to visualize")
        return

    try:
        # Load Harvard-Oxford atlas
        print("Loading Harvard-Oxford atlas...")
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_filename = atlas.maps
        atlas_labels = atlas.labels

        # Create a mapping from region names to atlas indices
        region_mapping = {}
        for i, label in enumerate(atlas_labels):
            # Clean up label names to match your data
            clean_label = label.replace('_', ' ').replace('-', ' ')
            region_mapping[clean_label] = i

        # Create statistical maps for each amygdala
        for seed in ['Amygdala_L', 'Amygdala_R']:
            seed_results = results_df[results_df['seed'] == seed].copy()

            if len(seed_results) == 0:
                print(f"No data for {seed}")
                continue

            # Filter for significant changes
            significant_results = seed_results[
                (seed_results['p_value'] < 0.05) &
                (seed_results['t_statistic'].abs() > 2.0)
                ].copy()

            if len(significant_results) == 0:
                print(f"No significant changes for {seed}, showing top 10 by t-statistic")
                significant_results = seed_results.nlargest(10, 't_statistic').copy()

            # Create statistical map
            stat_map = np.zeros(len(atlas_labels))

            for _, row in significant_results.iterrows():
                region_name = row['region']
                t_stat = row['t_statistic']

                # Try to find matching region in atlas
                found = False
                for atlas_label in atlas_labels:
                    if region_name.lower() in atlas_label.lower() or atlas_label.lower() in region_name.lower():
                        atlas_idx = atlas_labels.index(atlas_label)
                        stat_map[atlas_idx] = t_stat
                        found = True
                        break

                if not found:
                    print(f"Could not map region: {region_name}")

            # Create brain visualization
            if np.any(stat_map != 0):
                try:
                    # Create a NIfTI image from the statistical map
                    atlas_img = nib.load(atlas_filename)
                    stat_img = nib.Nifti1Image(stat_map.reshape(atlas_img.shape), atlas_img.affine)

                    # Create the plot
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle(f'{seed} Connectivity Changes (MRI3 vs MRI1)', fontsize=16)

                    # Axial view
                    plotting.plot_stat_map(stat_img, axes=axes[0, 0],
                                           title='Axial View', cut_coords=(0, 0, 0),
                                           colorbar=True, display_mode='z')

                    # Sagittal view
                    plotting.plot_stat_map(stat_img, axes=axes[0, 1],
                                           title='Sagittal View', cut_coords=(0, 0, 0),
                                           colorbar=True, display_mode='x')

                    # Coronal view
                    plotting.plot_stat_map(stat_img, axes=axes[1, 0],
                                           title='Coronal View', cut_coords=(0, 0, 0),
                                           colorbar=True, display_mode='y')

                    # Glass brain view
                    plotting.plot_glass_brain(stat_img, axes=axes[1, 1],
                                              title='Glass Brain View', colorbar=True)

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f'{seed.replace(" ", "_")}_brain_visualization.png'),
                                dpi=300, bbox_inches='tight')
                    plt.show()

                    print(f"Created brain visualization for {seed}")
                except Exception as e:
                    print(f"Error in brain plotting: {e}")
                    # Create a simple bar plot as fallback
                    fig, ax = plt.subplots(figsize=(12, 8))
                    significant_results_sorted = significant_results.sort_values('t_statistic', ascending=True)
                    colors = ['red' if x > 0 else 'blue' for x in significant_results_sorted['t_statistic']]
                    ax.barh(range(len(significant_results_sorted)), significant_results_sorted['t_statistic'],
                            color=colors, alpha=0.7)
                    ax.set_yticks(range(len(significant_results_sorted)))
                    ax.set_yticklabels([region[:30] + '...' if len(region) > 30 else region
                                        for region in significant_results_sorted['region']])
                    ax.set_xlabel('T-Statistic')
                    ax.set_title(f'{seed} - Brain Regions with Connectivity Changes')
                    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f'{seed.replace(" ", "_")}_connectivity_changes.png'),
                                dpi=300, bbox_inches='tight')
                    plt.show()
                    print(f"Created connectivity changes plot for {seed}")
            else:
                print(f"No valid regions found for {seed} visualization")

    except Exception as e:
        print(f"Error creating brain visualization: {e}")
        print("This might be due to missing nilearn or atlas data")


if __name__ == '__main__':

    print("=== Amygdala Seed Connectivity Analysis ===")

    # Get correlation matrices
    correlation_matrix = PearsonCorr(project_root)

    if not correlation_matrix:
        print("No correlation matrices created. Check your data files.")
        exit()

    # Export correlation matrices
    export_correlation_matrices(correlation_matrix, output_folder)

    # Extract amygdala correlations
    amygdala_correlations = extract_amygdala_correlations(correlation_matrix)
    between = between_group_tests(
        amygdala_correlations,
        randomization_xlsx=r"C:\Users\amirh\Downloads\RandomizationTable.xlsx",  # update path if needed
        output_folder=output_folder,
        subject_col="SubID",
        group_col="Group_Simbol"
    )
    volcano_plot(between, output_folder)
    # Analyze session changes
    results = analyze_session_changes(amygdala_correlations)

    if results is not None:
        plot_session_changes(results, output_folder)

        # Create brain visualizations
        print("\n=== Creating Brain Visualizations ===")
        create_brain_visualization(results, output_folder)
    else:
        print("Could not perform analysis")

    print("\n=== Analysis Complete ===")

