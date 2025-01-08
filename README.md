**GPS-DTI: An interpretable geometric graph neural network for drug–target interaction prediction**

**Overview**

GPS-DTI is a deep learning model designed to predict Drug-Target Interactions (DTIs). 

**Requirements**

Python Environment
Python 3.8.x (Tested with 3.8.19)
Key Libraries
To install the necessary dependencies, use the provided requirements.txt file:

pip install -r requirements.txt

Below are the key libraries and their versions:

torch==1.10.0
torch-geometric==2.0.3
ogb==1.3.0
scikit-learn==0.24.2
scipy==1.7.3
numpy==1.21.2
scikit-image==0.18.3

**Installation**

1.Clone the repository:

git clone https://github.com/xa-123955/GPS-DTI.git

cd GPS-DTI

2.Install the required libraries:

pip install -r requirements.txt

**Usage**

To train the model on your own dataset, follow these steps:

1. Prepare Your Data
Make sure your dataset is in the correct format with two columns: drug and target, and corresponding interaction labels.

2. Train the Model
Run the following command to start the training process:

python train.py --train-data <path_to_train_data> --epochs 50 --batch-size 32 --learning-rate 1e-4


**Results**

Results show that GPS-DTI outperforms existing models in terms of prediction accuracy, especially under cold-start conditions. The model is also capable of providing interpretable insights into drug-target interactions through the attention mechanism.
