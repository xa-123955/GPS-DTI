# GPS-DTI: An interpretable geometric graph neural network for enhancing the generalizability of drug–target interaction prediction
## Requirements


Below are the key libraries and their versions:

  Python==3.8.19
  
  torch==1.10.0

  torch-geometric==2.0.3

  ogb==1.3.0

  scikit-learn==0.24.2

  scipy==1.7.3

  numpy==1.21.2

  scikit-image==0.18.3

## Installation

1. create a new conda environment
```  
    conda create --name GPS-DTI python=3.8

    conda activate GPS-DTI
```
2. install requried python dependencies
```
    pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
     
    pip install torch-scatter==2.0.8 torch-sparse==0.6.13 torch-cluster==1.5.9 torch-spline-conv==1.2.1
   
    pip install torch-geometric==2.0.3
   
    conda install -c conda-forge rdkit==2021.03.2
   
    pip install ogb==1.3.0
   
    pip install -U scikit-learn
   
    pip install yacs
   
    pip install prettytable
```

3. Clone the repository:

    git clone https://github.com/xa-123955/GPS-DTI.git

    cd GPS-DTI

2. Install the required libraries:

    pip install -r requirements.txt

## Usage

To train the model on your own dataset, follow these steps:

1. Prepare Your Data
   
Make sure your dataset is in the correct format with two columns: drug and target, and corresponding interaction labels.

2. Train the Model

For the experiments with vanilla DrugBAN, you can directly run the following command. ${dataset} could either be bindingdb, biosnap ,C.elegans and human. ${split_task} could be random , cold and cluster.

Run the following command to start the training process:

    python main.py --cfg "configs/GPS-DTI.yaml" --data ${dataset} --split ${split_task}
    
## Directory Structure Description

```
filetree 
├── configs
├── datasets
├── README.md
├── configs.py
├── dataloder.py
├── main.py
├── models.py
├── trainer.py
└── util.py

```


## Results

Results show that GPS-DTI outperforms existing models in terms of prediction accuracy, especially under cold-start conditions. The model is also capable of providing interpretable insights into drug-target interactions through the attention mechanism.
