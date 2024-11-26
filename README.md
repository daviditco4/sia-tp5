# Task 1
This task contains an implementation of an autoencoder. 

## System requirements
* Python 3.7+


## Configuration
The Configuration file with the optimal parameters is `prototype.json` found in the configs folder. In this folder you'll find .json files with different configurations, varying:
* Encoder Layers
* Learning Rate
* Momentum
* Optimizer 


## How to use
 Clone or download this repository in the folder you desire
* In the runner_all.py file, you can select the configuration file you want to use.
* In a new terminal, navigate to the `sia-tp5` repository using `cd`
* When you are ready, enter a command as follows:
```sh
python3 scripts/runners/runner_all.py
```
* To plot the data, you can enter a command as follows:
```sh
python3 scripts/plotters/training_error_vs_epoch.py <data_csv_file> <varying_hyperparameter_name>
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.