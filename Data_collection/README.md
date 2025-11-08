# DeltaCata-DB
Data collection pipeline.
<img src="./DeltaCata_DB.svg">

## Requirements
To run the data collection pipeline, you also need to install the following additional packages (on top of the main project’s environment):
```
pip install requests
pip install zeep
pip install beautifulsoup4
pip install pubchempy
pip install numpy
pip install lxml
pip install html5lib
```

## Usage

Run the notebooks in order:

`01Downloading and preprocessing BRENDA data.ipynb`
Download BRENDA records and perform basic cleaning/normalization.

`02Downloading and preprocessing SABIO-RK data.ipynb`
Fetch SABIO-RK entries and apply the same preprocessing to keep schemas consistent.

`03Remove duplicates between databases.ipynb`
Merge the two datasets and remove duplicates according to the matching rules in the notebooks.

Tips:
Execute all cells in each notebook before moving to the next.
Ensure network access for external data sources (BRENDA, SABIO-RK).
Confirm the extra packages above are installed in your active environment.

## Large Language Model (LLM) and Human Review Assistance

We used the LLM to accurately parse buffer information for an enzymatic reaction from the unstructured free‑text “COMMENTARY” field in BRENDA. Specifically, we apply the **Llama 3.1 405B** model to extract buffer information from this unstructured text during the BRENDA stage, followed by careful human review for verification.

