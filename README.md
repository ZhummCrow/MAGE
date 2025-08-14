# MAGE
Enzyme optimization.

# Table of Contents

- [Local Installation](#local-installation)
   - [Requirements](#requirements)
   <!-- - [Installation](#installing) -->
   - [Prediction](#predict)
   - [Reproducibility](#reproduce)


## Local Installation <a name="local-installation"></a>

To run MAGE on a local machine, please follow the instructions below.

### Requirements <a name="requirements"></a>
```
conda create --name MAGE python=3.8
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

### Prediction <a name="predict"></a>
This project supports two inference modes:

1. User-specified mutations (following `input_json/example.json`).
2. Deep Mutational Scanning (following `input_json/example_dms.json`).

Both modes require a protein sequence, a ligand SMILES string, and the according PDB file path.

```
cd Inference

# Mode A: specified mutations (example.json)
python inference.py --input input_json/example.json

# Mode B: DMS (example_dms.json)
python inference.py --input input_json/example_dms.json
```

**Input json format:**
Required fields (both modes):

~ sequence: string; amino-acid sequence using one-letter codes.
~ ligand_smiles: string; SMILES of the ligand.
~ pdb_path: string; absolute or project-relative path to a reference PDB file.



### Reproducibility <a name="reproduce"></a>
Download the split and features xxxx.
```
cd Inference
python reproduce.py
```