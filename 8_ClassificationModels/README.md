This text file outlines the purpose of each .ipynb & .py file in this directory. 

DatasetClassification_gradient_accum.py -> This script replicates the ConveNext_Tiny Dataset classification logic used in the paper.
ResNet-50_Convnext-Tiny_Classification.py -> This script replicates the ResNet-50 gender classification logic used in the paper. It also allows for the use of ConveNext_Tiny however the other .py files is still the primary ConveNext_Tiny implementation.
LogisticRegression.py ->  This script replicates the LogisticRegression model used in the paper.
MLP.py -> This script replicates the MLP model used in the paper.
SentenceT5_Probe_PaperAligned.py ->  This script replicates the SentenceT5 probe used in the paper.

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Trains models probes to identify spuriosu features within the datasets.