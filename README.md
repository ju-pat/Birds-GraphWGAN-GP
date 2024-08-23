# Birds-GraphWGAN-GP


# GraphWGAN for seabirds central-place foraging trips Project

This repository contains code and data for the GraphWGAN central-place foraging trips Project, which implements a Graph-based Wasserstein Generative Adversarial Network with gradient penalty (WGAN-GP) model for generating realistic seabird trajectories.

## Repository Structure

### Important files and folders:

- **Main.ipynb**: Main document that serves as the main entry point for running the project, including some data visualization, model training, fake data generation and evaluation.

- **Data/**: Contains the pre-processed single trajectories (segmented into individual trajectories, completed and reshaped for 50 steps by trajectory)
- - the subfolders divide the data by year

- **GraphWGAN_GP1 - GraphWGAN_GP5/**: Various versions of the GraphWGAN-GP model scripts, each version corresponding to different experimental settings.

- **VAE_1.ipynb**: Jupyter Notebook implementing a Variational Autoencoder (VAE) for graph data, potentially used as a baseline or for comparison with the GraphWGAN model for future work.



### Notebooks used (not to be runned as the data used for them is not available)

- **Visual_data.ipynb**: for visualizing the dataset.

- **Segm des traj_dist.ipynb**: for segmenting GPS data into single trajectories.

- **Data completion.ipynb**: for completing the trajectories iwith missing values and ensuring a constant segmentation time.

- **Reshape_data.ipynb**: for reshaping the trajectories into the desired number of steps



### Tools files:

- **tools.py**: Functions for calculating trajectory properties

- **graph_visualisation.py**: Functions for visualizing the generated graphs and vectors, both from the real and fake dataset

- **to_graph_utils.py**: Functions for converting various data formats into graph structures and vice-versa.



### Other secondary files and folders:

- **Amedee_GAN**: the notebook of the model trained from Amedee Roy's article "Using generative adversarial networks (GAN) to simulate central-place foraging trajectories".

- **logs/**: Logs generated during model training and evaluation for TensorBoard.

- **GraphWGAN_GP1/2/3/4/**: contain pre-trained models and the figures of the results obtained



## Getting Started

1. Clone the repository and open `Main.ipynb`. For data visualization and pre-processing steps, see `Visual_data.ipynb`, `Segm des traj_dist.ipynb`, `Data_completion.ipynb`, and `Reshape_data.ipynb`.

2. In `Main.ipynb`, choose the model to be trained, and define the hyper-parameters (at least the number of epochs has to be modified).

3. If desired (the pre-runned script outputs remain available in this repo), execute the scripts in the `Main.ipynb`, visualize the real dataset and the generated trajectories.

4. You can follow the training using the tensorboard command written in the code and executing it in a terminal.

5. To use re-trained models, modify the part before training of generator and discriminator weight initialization.



## Contact

For any questions or issues, please contact [julien.patras@gmail.com].
