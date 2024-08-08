# MOBiceps
![MOBicpes logo](./images/MOBiceps.png)
MOBiceps is a collection of python functions for omics and mass spectrometry data. It is the working arm of [MOAgent](https://github.com/wollscheidlab/MOAgent). An early version of its core function featureSelector.py was first time applied in the work [Nature Communications, 2023](https://www.nature.com/articles/s41467-023-42101-z) to identify phenotype-specific proteins in myeloproliferative neoplasms (blood cancer).
If you have any questions please do not hesitate to contact jsettelmeier@ethz.ch

## Installation

## Dependencies

MOBicepcs requires:

- joblib==1.2.0
- numpy==1.21.6
- pandas==1.4.1
- seaborn==0.11.2
- yellowbrick==1.5
- scikit-learn==1.2.1
- scipy==1.8.1
- scikit-image==0.19.2
- matplotlib==3.5.1
- shap==0.41.0
- xgboost==1.6.2
- tqdm==4.63.0
- pyopenms==2.7.0
- numba==0.55.1
- torch==1.13.1
- torchvision==0.14.1
- torchaudio==0.13.1
- statsmodels==0.13.2

### User installation 

You can use MOBiceps by cloning the repo to your local machine using

```bash
$ git clone https://github.com/wollscheidlab/MOBiceps.git
```
or using the python package distribution system pip via

```bash
$ pip install MOBiceps
```

## Notes
Developed and tested under python 3.8. 


## User Guide for core functions:

### Convert Data

Within your code if you installed MOBiceps via pip

```bash
import MOBiceps as mob
mob.convertRAWMP(original_format,target_format,number_of_cores)
```

Or to execute the script, navigate to its location in your terminal and use the following command:

```bash
python /path/to/MOBiceps/convertRAWMP.py --p /path/to/folder --s original_format --f target_format --c number_of_cores
```
with following parameters:

- `--p` or `--path_to_folder`: Absolute path to the folder containing all files to be converted. (Default: Current working directory)
- `--s` or `--orig_format`: Source file format. (Default: 'raw')
- `--f` or `--file_format`: Target file format. (Default: 'mzML')
- `--c` or `--core_number`: Specifies the number of threads to be used to convert all files. -1 corresponds to all possible. (Default: -1)


### Construct Feature Table for Feature Selector using Spectronaut or DIA-NN search output

Within your code if you installed MOBiceps via pip

```bash
from MOBiceps.expression_table import create_rfe_expression_table
feature_expression_table = create_rfe_expression_table(path_to_search_output, path_to_class_annotation, path_to_output) 
```

Or to execute the script, navigate to its location in your terminal and use the following command:

```bash
python /path/to/MOBiceps/expression_table.py --s /path/to/search/output --c /path/to/class/annotation --o /path/to/output
```
with the following parameters:

- `--s`: Path to search output. Currently Spectronaut and DIA-NN output is supported. (Default: Current working directory)
- `--c`: Path to class annotation file. (Default: Current working directory)
- `--o`: Output path. (Default: Current working directory)
- `--m`: Imputation method. Currently "mean", "median", "zero", "gaussian" are supported. (Default: "none")
- `--f`: Feature level. "peptide" and "protein" are supported. (Default: "peptide")

### Apply RFE++ to derive predictive features

Within your code if you installed MOBiceps via pip

```bash
from MOBiceps.rfePlusPlusWF import execute_rfePP
most_contributing_features = execute_rfePP(path_to_search_output, path_to_class_annotation, path_to_output) 
```
Or to execute the script, navigate to its location in your terminal and use the following command:

```bash
python /path/to/MOBiceps/rfePlusPlusWF.py --i /path/to/search/output --c /path/to/class/annotation --o /path/to/output --p classA classB classC
```

with the following parameters:

- `--i`: Path to the folder containing the search output of Spectronaut or DIA-NN. (Default: Current working directory)
- `--c`: Path to the class annotation file.
- `--s`: Path to the sample annotation file. (Optional)
- `--o`: Output path. (Default: Current working directory)
- `--b`: Use bootstrapping augmentation. (Default: False)
- `--m`: Imputation method. Currently "mean", "median", "zero", "frequent" and "none" are supported. (Default: 'none')
- `--f`: Feature level. "peptide" and "protein" are supported. (Default: 'peptide')
- `--g`: Support for GPU if set to True. (Default: False)
- `--n`: Bootstrapping with noisy resampling. (Default: False)
- `--h`: Force the reduction to a handable  amount of features. (Default: True)
- `--p`: specify which classes should be considered. 

### Demo for the RFE++ 

The repo of MOBiceps comes with the metabolomics demo used in [MetaboAnalyst Tutorials](https://www.metaboanalyst.ca/docs/Tutorials.xhtml). We show here how to use the RFE++ function to derive the most predictive features to differentiate between Glomerulonephritis affected patients and healthy controls. 

```bash
from MOBiceps.rfePlusPlusWF import execute_rfePP

path_to_search_output = '.../Demo/input/metabolite_expression_table.csv'
path_to_class_annotation = '.../Demo/input/class_annotations.csv'
path_to_output = '.../Demo/output'

most_contributing_features = execute_rfePP(path_to_search_output, path_to_class_annotation, path_to_output) 
```
For the interpretation of the automatic created visualizations in the output folder, please consult our paper doi://to_be_added


