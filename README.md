# DeltaCata
Enzyme optimization.
<img src="./DeltaCata.jpg">

## 📖 Table of Contents
- [Local Installation](#local-installation)
   - [Requirements](#requirements)
   - [Inference](#inference)
   - [Reproducibility](#reproduce)
- [Data collection](#data-collection)
<!-- - [Acknowledgements](#acknw) -->
- [License](#license)
- [Citations](#citations)

## 💻 Local Installation <a name="local-installation"></a>

To run DeltaCata on a local machine, please follow the instructions below.

### 📦 Requirements <a name="requirements"></a>
```
conda create --name DeltaCata python=3.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas
pip install fair-esm
pip install tqdm
pip install biopython

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install torch-geometric

pip install rdkit
```


### 🔍 Inference <a name="inference"></a>
This project supports two inference modes:

1. User-specified mutations (following `input_json/example.json`).
2. Deep Mutational Scanning (following `input_json/example_dms.json`).

Both modes require a protein sequence, a ligand SMILES string, and the according PDB file path.

```
cd Inference/

# Mode A: specified mutations (example.json)
python inference.py --input input_json/example.json

# Mode B: DMS (example_dms.json)
python inference.py --input input_json/example_dms.json
```

**Input json format:**
Required fields (both modes):

* sequence: string; amino-acid sequence.
* SMILES: string; SMILES of the ligand.
* pdb_path: string; path to a reference PDB file.

### 🔁 Reproducibility <a name="reproduce"></a>
With the split in `Dataset/test_dataset/` and protein structures in `Dataset/test_pdbs/`, the following pipeline reproduces the reported results.
```
cd Inference
bash reproduce.bash
```


## 📊 Data Collection <a name="data-collection"></a>
We curate mutation-induced changes in enzyme kinetics (kcat and Km) primarily from the BRENDA and SABIO-RK databases. The complete pipeline for retrieval, curation, normalization, and dataset construction is provided in `Data_collection/`.

For quick use, ready-to-use datasets are available in `Dataset/` as `delta_kcat.csv` and `delta_km.csv`. 


<!-- ## 🙏 Acknowledgements <a name="acknw"></a>
xxx -->

## 📄 License <a name="license"></a>
This source code is licensed under the Attribution-NonCommercial-NoDerivatives 4.0 International license found in the `LICENSE` file in the root directory of this source tree.

## 🗞️ Citation and contact <a name="citations"></a>
Citation: Coming soon.


Contact:  
Qianmu Yuan (yuanqm3@mail3.sysu.edu.cn) 
Yuedong Yang (yangyd25@mail.sysu.edu.cn)