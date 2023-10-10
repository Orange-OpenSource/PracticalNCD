# A Practical Approach to Novel Class Discovery in Tabular Data

## Installation
```bash
# Create the virtual environment and install the packages with conda
conda env create --file environment.yml --prefix ./venvpracticalncd
# Activate the virtual environment
conda activate .\venvpracticalncd
# Add the virtual environment as a jupyter kernel
ipython kernel install --name "venvpracticalncd" --user
# Check if torch supports GPU (you need CUDA 11 installed)
python -c "import torch; print(torch.cuda.is_available())"
```


## Execution
Two notebooks are available:
- **Full_notebook.ipynb** illustrates how to run the models when the number of clusters *k* is known in advance.
- **Full_notebook_with_k_estimation.ipynb** (self-explanatory).


## Datasets
The datasets will be <u>automatically downloaded</u> from https://archive.ics.uci.edu/ on the first execution.<br/>
If it fails, please try disabling proxies.

**However**, the data splits for some datasets are random and the results can vary compared to the paper.

The most impacted datasets are:
- LetterRecognition
- USCensus1990
- multiple_feature


## Citation
If you found this work useful, please use the following citation:
```
...
```