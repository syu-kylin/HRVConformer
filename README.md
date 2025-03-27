# HRVConformer: Neonatal Hypoxic-Ischemic Encephalopathy Classification from HRV signal with Conformer Network

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/79462188/424516980-930c636d-0ee5-4229-867f-e19c5fbba9fe.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDIzOTM5MTMsIm5iZiI6MTc0MjM5MzYxMywicGF0aCI6Ii83OTQ2MjE4OC80MjQ1MTY5ODAtOTMwYzYzNmQtMGVlNS00MjI5LTg2N2YtZTE5YzVmYmJhOWZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAzMTklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMzE5VDE0MTMzM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI5ZTQ1NTZhZDY5Y2I0OGU3NzRlZmNhZTk2OTNlYzRlOTk2YTQ4MzRkNGQ5YjMwNTJiZmZmYzkyODk5NDIwNzgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.nzFMzI06TPUGsIyNAfBEy3A6C1EH8_iflmNtop2TRio" width="600">
</p>

This is PyTorch implementation of [HRVConformer paper]():
```bibtex
@article{your_paper_reference,
  author    = {Your Name and Others},
  title     = {HRVConformer: Neonatal Hypoxic-Ischemic Encephalopathy Classification from HRV signal with Conformer Network},
  journal   = {Journal Name},
  year      = {2025}
}
```
Please cite our paper if you found it useful.

## Install and Usage
1. **Clone the repository:**  
   ```bash
   git clone git@github.com:syu-kylin/HRVConformer.git
   cd HRVConformer
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your datasets:**
    Change your dataset path in `data_loader.py/read_split_data`, or add your own `Dataset` module.

4. **Run and Config model:**
   Model and training parameters have been configured in `project_init.py`, it also can be changed from here.
   **Launch training:**
   ```bash
   python train_main.py
   ```

