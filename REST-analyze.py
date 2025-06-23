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
# I WILL SEND YOU LIST OF CONFUNDS - FILTER JUST THE RELEVENT ONE AND MAKE SURE YOU REMOVE THE HEADLINES
# add also harward oxford subcortex atlas - analyze them togehter


STANDARTIZE = 'zscore'
SMOOTHING_FWHM = 4
DETREND = True
HIGH_PASS = 0.01
REGENERATE_TIME_SERIES = False  # NEW: boolean to control regeneration
T_R = 1
LOW_PASS = 0.08
DEBUG = True
project_root = r"C:\fmri_data"
csv_dir= ""

def RemoveFirstNVolumes(nifti, num_vol_to_remove):
    print('RemoveFirstNVolumes')
    img = nib.load(nifti)
    data = img.get_fdata()[:, :, :, num_vol_to_remove:]
    img_sliced = nib.Nifti1Image(data, img.affine, img.header)
    """img_sliced_path = ''.join(nifti.split('.')[0])+'_r.nii'
    nib.nifti1.save(img_sliced, img_sliced_path)"""
    return img_sliced

def GetAtlasAndLabels():
    global atlas_img, labels
    atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
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
                            if 'preproc_bold' in file and 'rest' in file and file.endswith('.nii.gz'):
                                bold_file = os.path.join(func_path, file)
                            if 'confounds' in file and 'rest' in file and file.endswith('.tsv'):
                                confounds_file = os.path.join(func_path, file)

                        if bold_file and confounds_file:
                            fs = FMRIFileSet(subject_folder, session_folder, bold_file, confounds_file)
                            file_sets.append(fs)

    return file_sets


def create_time_series():
    atlas_labels, atlas_img = GetAtlasAndLabels()
    file_sets = get_func_files(project_root)

    ts_dict = {}  # NEW: {(sub_id, ses_id): DataFrame}

    # create dictionary (subject, session) -> time_series
    for file_set in file_sets:
        nifti_sliced = RemoveFirstNVolumes(
            nifti=file_set.bold_path, num_vol_to_remove=4
        )
        conf_ = (
            pd.read_csv(file_set.confounds_path, sep="\t")
            .iloc[4:]
            .reset_index(drop=True)
        )

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
        df = pd.DataFrame(time_series)

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
