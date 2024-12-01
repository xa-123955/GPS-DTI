# GPS-DTI

**GPS-DTI** is a deep learning framework designed to accurately predict drug-target interactions (DTI), addressing the challenges of generalization in existing computational methods. By integrating advanced graph neural networks and attention mechanisms, GPS-DTI achieves state-of-the-art performance in both in-domain and cross-domain prediction tasks.

---

## Features

- **Hybrid Architecture**: Combines a GINE graph neural network with a multi-head attention mechanism to capture both local and global features of drug molecules.
- **Advanced Protein Representations**: Leverages a pre-trained ESM-2 model and CNN-based refinement for enhanced protein feature extraction.
- **Cross-Attention Module**: Enables detailed analysis of interactions between drug substructures and protein amino acid sequences, enhancing biological interpretability.
- **Superior Performance**: Outperforms benchmark models on in-domain, cross-domain, and drug-target affinity (DTA) prediction tasks, including applications related to COVID-19.
- **Interpretable Insights**: Provides visualized cross-attention maps to aid understanding of drug-target interactions.

---

## Installation and Usage

```bash
# Step 1: Clone the Repository
git clone https://github.com/xa-123955/GPS-DTI.git
cd GPS-DTI

# Step 2: Install Dependencies
conda env create -f environment.yml
conda activate gps-dti

# Step 3: Training the Model
python scripts/train_model.py --config config/config.ini

