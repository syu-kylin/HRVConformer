# Data access and expected layout

## Availability

The clinical ECG/HRV data used in the paper originate from the multicentre ANSeR1 and ANSeR2 cohort studies (ClinicalTrials.gov identifiers [NCT02160171](https://clinicaltrials.gov/study/NCT02160171) and [NCT02431780](https://clinicaltrials.gov/study/NCT02431780)). They contain sensitive participant data and are not redistributed in this repository.

Access is subject to the original ethics approvals, consent arrangements, institutional governance, and data-sharing agreements. Opening an issue in this repository does not constitute a data-access request and cannot grant access.

## Study split represented by the loader

- Development: ANSeR2 expert-labelled epochs plus eligible weakly labelled intermediate epochs.
- Validation: 20% of one-hour development epochs, sampled with `--seed_epoch`.
- Independent test: ANSeR1 expert-labelled epochs.
- Binary target: normal/mild EEG grade (class 0) versus moderate/severe/inactive grade (class 1).

The current implementation reproduces the paper's epoch-level development split. It is not a patient-grouped cross-validation implementation. The independent test cohort comes from ANSeR1.

## Preprocessed file layout

Set `HRVCONFORMER_DATA_ROOT` or pass `--data_root` so it points to the directory containing the window-length folders:

```text
RPeaks/
└── 5mins/
    ├── NN_epoch_ANSeR2_weak_7-11h_5min_std_0.12.mat
    ├── NN_epoch_ANSeR2_weak_13-23h_5min_std_0.12.mat
    ├── NN_epoch_ANSeR2_weak_25-35h_5min_std_0.12.mat
    ├── NN_epoch_ANSeR2_weak_37-47h_5min_std_0.12.mat
    ├── NN_epoch_ANSeR2_strong_6-48h_5min_std_0.12.mat
    └── NN_epoch_ANSeR1_strong_6-48h_5min_std_0.12.mat
```

Each MATLAB file must contain `NN_epoch_ANSeR`, whose records provide:

| Field | Meaning |
|---|---|
| `file_id` | Stable one-hour epoch identifier; its prefix identifies the infant |
| `rr_epochs` | Fixed-length, interpolated NN-interval windows |
| `EEG_grade` | Binary class label (`0` or `1`) |
| `n_epochs` | Number of windows belonging to the one-hour epoch |

For the paper configuration, each `rr_epochs` row has 1,200 samples (five minutes at 4 Hz). The preprocessing described in the manuscript excludes noisy windows with NN-interval standard deviation above 0.12 s and one-hour epochs with fewer than ten retained windows.

## Privacy and repository hygiene

Keep all raw and derived participant data outside the repository. Do not commit participant identifiers, clinical metadata, signals, model outputs keyed by participant, or trained weights that may require disclosure review. The supplied `.gitignore` excludes common data and checkpoint formats as a secondary safeguard, not as a substitute for data governance.
