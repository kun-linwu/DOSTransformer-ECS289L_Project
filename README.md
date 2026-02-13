# Density of States Prediction of Crystalline Materials via Prompt-guided Multi-Modal Transformer
The offical source code for Density of States Prediction of Crystalline Materials via Prompt-guided Multi-Modal Transformer, accepted at 2023 Neurips main conference.
This project introduces DOSTransformer, a novel deep learning architecture designed to predict the Density of States (DOS) for crystalline materials by integrating crystal structures and energy levels as heterogeneous information. Accepted at the 2023 NeurIPS main conference, the model uses a multimodal transformer to capture complex relationships between atoms and energy states rather than focusing solely on material representations. By utilizing system-specific prompts, the framework effectively guides the model to learn structural interactions for both Phonon and Electron DOS across various real-world scenarios.
 
## Technical Skills Used
- Deep Learning Architectures: Implementation of Multimodal Transformers to fuse heterogeneous data sources.
- Graph Neural Networks (GNNs): Developing GNN-based embedders to extract high-quality representations from crystalline structures.
- Physics-Informed ML: Modeling spectral properties (DOS) by treating energy levels as active model inputs rather than static targets.
- Prompt Engineering: Using learnable prompts to guide structural system-specific interactions within the transformer layers.
- Data Engineering: Processing raw data from the [Materials Project](https://materialsproject.org/), converting formats to .pkl, and generating graph datasets via mat2graph.py.

## Features
The core feature of this project is the DOSTransformer architecture, which shifts the focus of DOS prediction from simple material representation to modeling the functional relationship between materials and energy levels. It supports two distinct physical domains: Phonon DOS, which provides insights into lattice vibrations, and Electron DOS, which is fundamental to understanding the electronic properties of materials.

The repository includes specialized modules for each task, such as DOSTransformer_phonon.py and DOSTransformer.py, alongside comprehensive training scripts. A flexible hyperparameter system allows users to tune the balancing term (--beta) for system-specific RMSE, adjust the number of transformer layers, and select various embedders to suit different material datasets.

### Phonon DOS Prediction
#### Dataset
You can dowload phonon dataset in this [repository](https://github.com/ninarina12/phononDoS_tutorial)  
#### Run model
Run `main_phDOS.py` for phonon DOS Prediction after downloading phonon DOS dataset into `data/processed`
### Electron DOS Prediction
#### Dataset
We build Electron DOS dataset consists of the materials and its electron DOS information which are collected from [Materials Project](https://materialsproject.org/)  
We converted raw files to `pkl` and made electronic DOS dataset by `mat2graph.py`  
#### Run model
Run `main_eDOS.py` for electron DOS Prediction after building electron DOS dataset.   
### Models
#### embedder eDOS
`DOSTransformer.py`: Our proposed model: DOSTransformer for Electron DOS
#### embedder phDOS
`DOSTransformer_phonon.py`: Our proposed model: DOSTransformer for Phonon DOS  

### Hyperparameters  
`--beta:` Hyperparameter for training loss controlling system_rmse (Balancing Term for Training)
`--layers:` Number of GNN layers in DOSTransformer model  
`--attn_drop:` Dropout ratio of attention weights
`--transformer:` Number of Transformer layer in DOSTransformer   
`--embedder:` Selecting embedder   
`--hidden:` Size of hidden dimension
`--epochs:`  Number of epochs for training the model
`--lr:` Learning rate for training the model  
`--dataset:` Selecting dataset for eDOS prediction (Random split, Crystal OOD, Element OOD, default dataset is Random split)
`--es:` Early Stopping Criteria  
`--eval:` Evaluation Step  

## The Process
The workflow began with addressing a gap in previous research: the neglect of energy levels as a primary determinant of DOS distribution. We developed a multimodal approach that treats the crystalline material and energy levels as two distinct modalities that interact within a transformer block. To enhance this, we integrated prompt-guided learning, allowing the model to adapt its predictions based on the specific crystal structural system being analyzed.

For the data pipeline, we curated datasets from the Materials Project and other specialized repositories, ensuring the materials were properly converted into graph-based representations. The model was then subjected to extensive testing, using Early Stopping Criteria and specific evaluation steps to ensure robust performance even in OOD scenarios where the crystal structures or chemical elements were unseen during training.

## What I've Learned
This project deepened my understanding of how Multimodal Transformers can be adapted for physical sciences, specifically for spectral properties that are inherently continuous. I learned that the traditional approach of "representation learning" is often insufficient for properties like DOS, where the interaction between the material and the probe (in this case, energy levels) is just as important as the material itself.

Working on a project accepted at NeurIPS also refined my ability to conduct rigorous OOD evaluations. Understanding how a model generalizes to new elements or crystal systems is critical for the real-world application of AI in material discovery, as it ensures the model is not just memorizing known structures but actually learning the underlying physics.

## Future Improvements
- Cross-Domain Pre-training: Implementing self-supervised pre-training on larger unlabeled material datasets to improve the GNN embedder's initial representations.
- Attention Visualization: Developing tools to visualize the transformer's attention maps to interpret which atoms contribute most to specific energy peaks.
- Scalability: Optimizing the transformer layers to handle larger unit cells with hundreds of atoms without a significant increase in computational cost.
- Integration with Active Learning: Connecting the DOS prediction pipeline to an active learning loop for autonomous discovery of materials with specific bandgap properties.

## Figures
![DOSTransformer_model_img](https://github.com/HeewoongNoh/DOSTransformer/assets/62690984/ae69a43a-20fd-4038-92b3-12938feacc8e)

(https://pure.kaist.ac.kr/en/publications/density-of-states-prediction-of-crystalline-materials-via-prompt-/)
