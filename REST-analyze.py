# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:17:09 2023

@author: marko
"""

import os
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
import nibabel as nib


STANDARTIZE = 'zscore'
SMOOTHING_FWHM = 4
DETREND = True
HIGH_PASS = 0.01
REGENERATE_TIME_SERIES = True  # NEW: boolean to control regeneration
T_R = 1
LOW_PASS = 0.08
DEBUG = True
project_root = r"C:\Users\USER\Desktop\לימודים\רפואה\מעבדה\KPE\subX"


def RemoveFirstNVolumes(nifti, num_vol_to_remove):
    img = nib.load(nifti)
    data = img.get_fdata()[:, :, :, num_vol_to_remove:]
    img_sliced = nib.Nifti1Image(data, img.affine, img.header)
    """img_sliced_path = ''.join(nifti.split('.')[0])+'_r.nii'
    nib.nifti1.save(img_sliced, img_sliced_path)"""
    return img_sliced


def GetAtlasAndLabels(atlas_type):
    atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford(atlas_type)
    atlas_img = atlas_harvard_oxford.maps
    atlas_labels = atlas_harvard_oxford.labels

    return atlas_labels, atlas_img


class FMRIFileSet:
    def __init__(self, subject, session, bold_path, confounds_path):
        self.subject = subject
        self.session = session
        self.bold_path = bold_path
        self.confounds_path = confounds_path

    def print_info(self):
        print(f"Subject: {self.subject}, Session: {self.session}")
        print(f"BOLD: {self.bold_path}")
        print(f"Confounds: {self.confounds_path}")


def get_func_files(project_root):
    file_sets = []

    for subject_folder in os.listdir(project_root):
        if subject_folder.startswith('sub-'):
            subject_path = os.path.join(project_root, subject_folder)

            for session_folder in os.listdir(subject_path):
                if session_folder.startswith('ses-'):
                    func_path = os.path.join(subject_path, session_folder, 'func')

                    if os.path.isdir(func_path):
                        bold_file = None
                        confounds_file = None

                        for file in os.listdir(func_path):
                            if 'preproc_bold' in file and 'rest' in file and 'MNI' in file and file.endswith('.nii.gz'):
                                bold_file = os.path.join(func_path, file)
                            if 'confounds' in file and 'rest' in file and file.endswith('.tsv'):
                                confounds_file = os.path.join(func_path, file)

                        if bold_file and confounds_file:
                            fs = FMRIFileSet(subject_folder, session_folder, bold_file, confounds_file)
                            file_sets.append(fs)

    return file_sets


def CreateDataFrame(atlas_labels, atlas_img, nifti_sliced, conf_):  # NEW FUNC
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        labels=atlas_labels,
        standardize=STANDARTIZE,
        memory="nilearn_cache",
        verbose=0,
        smoothing_fwhm=SMOOTHING_FWHM,
        detrend=DETREND,
        low_pass=LOW_PASS,
        high_pass=HIGH_PASS,
        t_r=T_R,
    )
    print(
        f"standardize: {STANDARTIZE}, smoothing_fwhm: {SMOOTHING_FWHM}, "
        f"detrend: {DETREND}, low_pass: {LOW_PASS}, high_pass: {HIGH_PASS}, t_r: {T_R}"
    )

    time_series = masker.fit_transform(nifti_sliced, confounds=conf_)
    # Convert to DataFrame with region names (skipping the 'Background' if needed)
    region_names = atlas_labels[1:] if atlas_labels[0] == 'Background' else atlas_labels
    df = pd.DataFrame(time_series, columns=region_names)

    return df


def create_time_series():
    cort_labels, cort_img = GetAtlasAndLabels('cort-maxprob-thr25-2mm')
    sub_labels, sub_img = GetAtlasAndLabels('sub-maxprob-thr25-2mm')
    file_sets = get_func_files(project_root)

    ts_dict = {}  # NEW: {(sub_id, ses_id): DataFrame}

    # create dictionary (subject, session) -> time_series
    for file_set in file_sets:
        nifti_sliced = RemoveFirstNVolumes(
            nifti=file_set.bold_path, num_vol_to_remove=4
        )
        confound_columns = [
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
            "std_dvars", "framewise_displacement",
            "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
            "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05"
        ]
        conf_df = pd.read_csv(file_set.confounds_path, sep="\t")
        conf_ = conf_df[confound_columns].iloc[4:].reset_index(drop=True)

        cort_df = CreateDataFrame(cort_labels, cort_img, nifti_sliced, conf_)  # NEW
        sub_df = CreateDataFrame(sub_labels, sub_img, nifti_sliced, conf_)  # NEW
        df = pd.concat([cort_df, sub_df], axis=1)

        # save the CSV as before
        output_file = f"{file_set.subject}_{file_set.session}_time_series.csv"
        df.to_csv(output_file, index=False)

        # NEW: store in dictionary
        ts_dict[(file_set.subject, file_set.session)] = df

    return ts_dict  # NEW: return the dictionary


def load_files_from_disk(csv_dir="."):
    ts_dict = {}
    for fname in os.listdir(csv_dir):
        if fname.endswith('_time_series.csv'):
            try:
                subj, sess, *_ = fname.split('_')
                path = os.path.join(csv_dir, fname)
                df = pd.read_csv(path)

                ts_dict[(subj, sess)] = df
                print(f"Loaded {fname} → key=({subj}, {sess}), shape={df.shape}")

            except Exception as e:
                print(f"Skipping {fname}: {e}")

    return ts_dict


if __name__ == '__main__':
    # get bool parameter from the cmd  generate_time_series
    if (REGENERATE_TIME_SERIES):
        print("Regenerating time series...")
        ts_dict = create_time_series()
    else:
        ts_dict = load_files_from_disk()

        import os
        import pandas as pd
        from scipy.stats import ttest_rel

        project_root = r"C:\Users\USER\Desktop\לימודים\רפואה\מעבדה\KPE\140725-20250714T170127Z-1-001\140725"
        output_folder = os.path.join(os.path.dirname(project_root), "t_test_results")
        os.makedirs(output_folder, exist_ok=True)


        def PearsonCorr(project_root):
            correlation_matrix = {}

            for ts_file in os.listdir(project_root):
                file_path = os.path.join(project_root, ts_file)
                df = pd.read_csv(file_path)
                corr = df.corr()
                correlation_matrix[ts_file] = corr

            return correlation_matrix


        def ROI_Connectivity(correlation_matrix):
            data = []

            for file_name, df in correlation_matrix.items():
                lA_lH = df.loc["Left Hippocampus", "Left Amygdala"]
                rA_rH = df.loc["Right Hippocampus", "Right Amygdala"]
                lA_vmPFC = df.loc["Frontal Medial Cortex", "Left Amygdala"]
                rA_vmPFC = df.loc["Frontal Medial Cortex", "Right Amygdala"]

                parts = file_name.replace(".csv", "").split("_")
                subject = parts[0]  # "sub-X"
                session = parts[1]  # "ses-MRI1"

                data.append({
                    "subject": subject,
                    "session": session,
                    "lA_lH": lA_lH,
                    "rA_rH": rA_rH,
                    "lA_vmPFC": lA_vmPFC,
                    "rA_vmPFC": rA_vmPFC,
                })

            ROI_connectivity_df = pd.DataFrame(data)
            return ROI_connectivity_df


        def T_test(df):
            roi_columns = ["lA_lH", "rA_rH", "lA_vmPFC", "rA_vmPFC"]
            results = []

            for roi in roi_columns:
                # --- MRI1 vs MRI2 ---
                df1 = df[df["session"] == "ses-MRI1"]
                df2 = df[df["session"] == "ses-MRI2"]
                df3 = df[df["session"] == "ses-MRI3"]

                common_subs_12 = set(df1["subject"]).intersection(df2["subject"])
                d1 = df1[df1["subject"].isin(common_subs_12)].sort_values("subject")
                d2 = df2[df2["subject"].isin(common_subs_12)].sort_values("subject")

                if len(common_subs_12) >= 5:
                    t_stat, p_val = ttest_rel(d1[roi], d2[roi])
                    results.append({
                        "ROI": roi,
                        "session1": "ses-MRI1",
                        "session2": "ses-MRI2",
                        "n_subjects": len(common_subs_12),
                        "t_stat": t_stat,
                        "p_value": p_val
                    })

                # --- MRI1 vs MRI3 ---
                common_subs_13 = set(df1["subject"]).intersection(df3["subject"])
                d1b = df1[df1["subject"].isin(common_subs_13)].sort_values("subject")
                d3 = df3[df3["subject"].isin(common_subs_13)].sort_values("subject")

                if len(common_subs_13) >= 5:
                    t_stat, p_val = ttest_rel(d1b[roi], d3[roi])
                    results.append({
                        "ROI": roi,
                        "session1": "ses-MRI1",
                        "session2": "ses-MRI3",
                        "n_subjects": len(common_subs_13),
                        "t_stat": t_stat,
                        "p_value": p_val
                    })

            return pd.DataFrame(results)


        if __name__ == '__main__':
            Correlation_Matrix = PearsonCorr(project_root)
            ROI_DataFrame = ROI_Connectivity(Correlation_Matrix)
            T_test_results = T_test(ROI_DataFrame)
            output_csv = os.path.join(output_folder, "t test results.csv")
            T_test_results.to_csv(output_csv, index=False)
            print(f"results in: {output_csv}")






