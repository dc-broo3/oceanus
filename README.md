# oceanus :trident: 


## Installation instructions :construction_worker:

This repository relies on the MW--LMC basis function expansion potential found at: https://github.com/sophialilleengen/mwlmc.
To install these simulations do as follows:

```bash
$ git clone --recursive https://github.com/sophialilleengen/mwlmc.git
$ pip install ./mwlmc 
```

Installing this repository can then be done as:
```bash
$ git clone --recursive https://github.com/dc-broo3/oceanus.git
```


## Virtual environment instructions :computer:

On Rusty, a virtual environment ready to use to run the stream generation code can be found at:

```bash
$ source /mnt/home/rbrooks/ceph/venvs/mwlmc_fulldiscexp_venv/bin/activate
```
![welcome plot](https://github.com/dc-broo3/oceanus/blob/8ae59dc931d4c5693507533af4493af27832f8a6/analysis/figures/plot_stream_coords.png)