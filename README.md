# M2REMAP
This repository contains the code and data for "Multimodal Representation Learning for Predicting Molecule-Disease Relations"


Predicting molecule-disease indications and side effects is important for drug development and pharmacovigilance. Comprehensively mining molecule-molecule, molecule-disease, and disease-
disease semantic dependencies can potentially improve prediction performance.

We introduce a Multi-Modal REpresentation Mapping Approach to Predicting molecular-  disease relations (M2REMAP) by incorporating clinical semantics learned from electronic health records
(EHR) of 12.6 million patients. Specifically, M2REMAP first learns a multimodal molecule representation  that synthesizes chemical property and clinical semantic information by mapping molecule chemicals
via a deep neural network onto the clinical semantic embedding space shared by drugs, diseases, and  other common clinical concepts. To infer molecule-disease relations, M2REMAP combines multimodal
molecule representation and disease semantic embedding to jointly infer indications and side effects.

Currently, the codes are provided only for a reference and cannot run directly due the the lack of the EHR embedding vecotrs, which will be published later.

![img.png](img.png)


# Requirements
* python 3.7
* tensorflow==2.3.0
* numpy >= 1.19
* pandas >= 1.3
* scikit-learn >= 1.0.2

# Usage
```sh
  1. first install the environment:  setup.sh
  2. side effect model learning: run_SIDER4.sh
  3. indication model training: run_indication.sh
```
