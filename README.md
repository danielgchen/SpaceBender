# SpaceBender
## Summary
<p>
    <img src="static/logo.png" alt="SpaceBender logo of purple cell looping around the earth" width="90" align="left"/>
    SpaceBender denoises spatial transcriptomics data in a spatially-aware
    <br>
    fashion and is based on existing single-cell denoising architecture.
    <br>
    See our manuscript at <a href="google.com">bioRxiv</a> for more information.
</p>

## Quick Usage
Example Python code to run SpaceBender, e.g. in a Jupyter notebook:
```
# import SpaceBender package, attach to path via "sys" package
import sys
sys.path.insert(0, '/path/to/<PACKAGE>')
from <PACKAGE> import spacebender

# define the paths of the input and output H5AD files
input_file = '/path/to/data/NAME.h5ad'
output_file = input_file.replace('.h5ad','.OUT.h5ad')

# run SpaceBender on the input H5AD file
spacebender.run_spacebender(
    input_file=input_file,
    output_file=output_file,
    k_neighbors=<N_NEIGHBORS, e.g. 6 or None (if using distance-based method)>,
    ambient_radius=<RADIUS, e.g. 100>,
    n_epochs=<N_EPOCHS, e.g. 100>,
    batch_size=<BATCH_SIZE, e.g. 256>,
    learning_rate=<LEARNING_RATE, e.g. 1e-3>,
    z_dim=<HIDDEN_DIMENSIONS, e.g. 64>,
    n_posterior_samples=<N_SAMPLES, e.g. 25>,
    device=<DEVICE, e.g. 'cpu' or 'cuda'>,
    train_in_tissue_only=False,
    random_seed=0)
```

Example Python code to run SpaceBender on Google CoLab:
```
# install packages
!pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
!pip install pyro-ppl scanpy tables seaborn matplotlib
!git clone https://github.com/danielgchen/SpaceBender.git

# import local SpaceBender package
from SpaceBender import spacebender
# upload your H5AD file then run!
input_file = '<EXAMPLE_H5AD_FILENAME>'
output_file = input_file.replace('.h5ad','.OUT.h5ad') # rename however you wish!
spacebender.run_spacebender(
    input_file=input_file,
    output_file=output_file,
    k_neighbors=<N_NEIGHBORS, e.g. 6 or None (if using distance-based method)>,
    ambient_radius=<RADIUS, e.g. 100>,
    n_epochs=<N_EPOCHS, e.g. 100>,
    batch_size=<BATCH_SIZE, e.g. 256>,
    learning_rate=<LEARNING_RATE, e.g. 1e-3>,
    z_dim=<HIDDEN_DIMENSIONS, e.g. 64>,
    n_posterior_samples=<N_SAMPLES, e.g. 25>,
    device=<DEVICE, e.g. 'cpu' or 'cuda'>,
    train_in_tissue_only=False,
    random_seed=0)
```

## Licensure
The original, in vast majority unmodified, CellBender code is located in the `remove_background`  and is under its original BSD-3 license. SpaceBender is under a BSD-3 Clear license in the top-level directory.

## Contact Us
Always happy to answer any questions to help improve and iterate upon this code!
Daniel G. Chen (email: dgchen@mednet.ucla.edu)
