# MAGE
Enzyme optimization.

# Table of Contents

- [Local Installation](#local-installation)
   - [System Requirements](#requirements)
   - [Installation](#installing)
   - [Prediction](#predict)
   - [Reproducibility](#reproduce)


## Local Installation <a name="local-installation"></a>

To run MAGE on a local machine, please follow the instructions below.

### 🖥️ System Requirements <a name="requirements"></a>
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

# ToDo

1. process dataset
2. data preparation code