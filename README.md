# Flexible task abstractions emerge in linear networks with fast and bounded units
by Kai J. Sandbrink*, Jan P. Bauer*, Alexandra M. Proca*, Andrew M. Saxe, Christopher Summerfield, Ali Hummos*<br/>
(* - equal contribution, randomized order)

Code for [Flexible task abstractions emerge in linear networks with fast and bounded units](https://openreview.net/forum?id=AbTpJl7vN6). For any questions about the code, contact [Alexandra](a.proca22@imperial.ac.uk) or [Jan](j.bauer@mail.huji.ac.il).

## Setup
To set up the conda environment, run:
```
conda env create -f NTA_environment.yml
conda activate NTA_environment
```

## Code
TODO: include a more comprehensive description of the repo<br/>
To run new simulations, you can create new configurations of hyperparameters (```Config``` in ```lcs/configs.py```) and use ```run_script.py```.

## Figures
In ```lcs/figure_notebooks/```, we include notebooks to reproduce all simulations and figures in the manuscript (labeled by the respective figure).

## Citation
Please cite our paper if you use this code in your research project.

```
@article{NTA2024,
  author = {Sandbrink, Kai J. and Bauer, Jan P. and Proca, Alexandra M. and Saxe, Andrew M. and Summerfield, Christopher and Hummos, Ali},
  title = {Flexible task abstractions emerge in linear networks with fast and bounded units},
  publisher = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```

