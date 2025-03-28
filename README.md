# HRVConformer: Neonatal Hypoxic-Ischemic Encephalopathy Classification from HRV signal with Conformer Network

<p align="center">
  <img src="https://s3.bmp.ovh/imgs/2025/03/28/35d76b114fd45a30.png" width="600">
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
The enhanced version of Pan-Tompkins algorithm can be seen at: [enhanced Pan-Tompkins](https://github.com/syu-kylin/enhanced-Pan-Tompkin).
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
    Change your dataset path in `data_loader.py/read_split_data()`, or add your own `Dataset` module.

4. **Run and Config model:**
   Model and training parameters have been configured in `project_init.py`, it also can be changed from here.
   **Launch training:**
   ```bash
   python train_main.py
   ```

