
# Stimulus symmetries can confound representational similarity analyses

This folder contains the codes and the associated data to replicate the main findings of the paper:
> **Stimulus symmetries can confound representational similarity analyses**, 
> Authors: Farhad Pashakhanloo and Jacob A. Zavatone-Veth
<a href="https://arxiv.org/abs/2605.21324" target="_blank" rel="noopener noreferrer">https://arxiv.org/abs/2605.21324</a>

There are two main jupyter notebooks that can be executed as standlone Python scripts:

- `RunToyModels.ipynb`: This script replicates all the results for the toy model. The figures are saved under `output/` folder.

- `RunRealData.ipynb`: This script replicates all the results related to the section with latent symmetry in image data . The saved neural network models are currently under `models/` folder. However, new models can be trained and saved (use appropriate flags in the code to enable this).



These codes were tested with:

- python 3.10.18
- numpy: 1.26.4
- torch: 2.8.0
- matplotlib: 3.10.5
- scikit-learn: 1.7.1
- scipy: 1.15.2
- Pillow (PIL): 11.3.0



