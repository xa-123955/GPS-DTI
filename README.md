GPS-DTI: An interpretable geometric graph neural network for drug–target interaction prediction

Overview
GPS-DTI is a deep learning model designed to predict Drug-Target Interactions (DTIs). The model integrates a GINE (Graph Isomorphism Network) graph neural network with a multi-head attention mechanism, aiming to enhance prediction accuracy and biological interpretability. It uses pre-trained protein embeddings from ESM-2 and includes a cross-attention module to capture both local and global features of drug molecules.

Requirements
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

Installation
1.Clone the repository:

git clone https://github.com/yourusername/GPS-DTI.git
cd GPS-DTI

2.Install the required libraries:

pip install -r requirements.txt

Model Architecture

Graph Neural Network (GINE): Used to extract drug and target features from molecular graphs, improving prediction accuracy by capturing the structure of the molecules.
Multi-head Attention Mechanism: Allows the model to focus on various parts of the graph representations, enhancing the ability to learn both local and global features.
Cross-Attention Module: Incorporates protein features via the ESM-2 pre-trained model, improving the biological interpretability of predictions.

Usage

To train the model on your own dataset, follow these steps:

1. Prepare Your Data
Make sure your dataset is in the correct format with two columns: drug and target, and corresponding interaction labels.

2. Train the Model
Run the following command to start the training process:

python train.py --train-data <path_to_train_data> --epochs 50 --batch-size 32 --learning-rate 1e-4

3. Evaluate the Model
After training, you can evaluate the model on a test dataset:

python evaluate.py --test-data <path_to_test_data> --model <path_to_trained_model>

Results
Results show that GPS-DTI outperforms existing models in terms of prediction accuracy, especially under cold-start conditions. The model is also capable of providing interpretable insights into drug-target interactions through the attention mechanism.
