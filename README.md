# MoEGP
MoEGP: an accelerating crop breeding approach based on the mixture of experts network

# Requirements
MoEGP is being developed on Linux and supports both GPU and CPU.

The code is written in Python 3 (>= 3.10). The list of dependencies can be found in the repository (see requirements.txt).

# Installation
1. Clone this Git repository.
2. Install a compatible version of Python, for example with Conda (other virtual environments like Docker, pyenv, or the system package manager work too).
   
   ```shell
    conda create --name MoEGP python=3.10
    conda activate MoEGP
   ```
3. Install requirements
   ```shell
    pip install -r requirements.txt
    ```
# Usage 
   MoEGP training and test using the following command.
   ```shell
   python run.py --input_dir data/processed --input_json input_data.json --output_dir model/MoEGP 
   --epochs 150 --batch_size 32 --lr 0.0001 0.00001 --dropouts 0.1 0.2 0.3 0.4 0.5 
   --hidden_dim_list [512] [1024, 512] --num_experts 6 10 --best_metrics pearson --zscore True
```
   The params of run.py is as following:
   ```shell
  --input_dir INPUT_DIR, input data dir, the data is already properly divided into training and validation sets, formatted as a data dictionary.
  --input_json INPUT_JSON, input json file, format as input_data.json
  --output_dir OUTPUT_DIR 
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR, learning rate 
  --dropouts DROPOUTS [DROPOUTS ...]
  --hidden_dim_list HIDDEN_DIM_LIST [HIDDEN_DIM_LIST ...]
  --num_experts NUM_EXPERTS [NUM_EXPERTS ...], the number of expert
  --best_metrics {pearson,rmse}
                        select the best model through metrics, default pearson
  --zscore ZSCORE       whether to use zscore normalization in the modeling
```
   The output dictionary is as following:

   ```shell
   .
   ├── metrics.csv  # metrics result from all task
   ├── TA-LS2013  # task1
   └── TN-LS2013  # task2
       └── TN-LS2013_Pred.csv  # predict result
```
   