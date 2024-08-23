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

1. Clone the repository and open `Main.ipynb`. For data visualization and pre-processing steps, see `Visual_data.ipynb`, `Segm des traj_dist.ipynb`, `Data_completion.ipynb`, and `Reshape_data.ipynb` (do not run them since they use other datasets).

2. In `Main.ipynb`, choose the model to be trained: modify the gan = "" and the True/False value for the corresponding WGAN-GP.

3. Since the pre-runned script outputs remain available in this repo, you can execute the scripts in the `Main.ipynb`, visualize the real dataset and the generated trajectories. By default, pre-trained models will be shown and not trained again (num_epochs=0). To train a model from scratch, comment the lines to load the model weights in the last cell of `Generator and Discriminator initialization`.

4. You can follow the training using the tensorboard command written in the code `tensorboard --logdir=logs` and executing it in a terminal.




## Contact

For any questions or issues, please contact [julien.patras@gmail.com].
