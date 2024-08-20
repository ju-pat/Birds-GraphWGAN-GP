# Birds-GraphWGAN-GP


# GraphWGAN for seabirds central-place foraging trips Project

This repository contains code and data for the GraphWGAN central-place foraging trips Project, which implements a Graph-based Wasserstein Generative Adversarial Network (WGAN) model for generating realistic seabird trajectories.

## Repository Structure

- **Data_preprocessed/**: Contains the pre-processed data of : segmented into individual trajectories, completed and reshaped for 50 steps by trajectory
- -
- -

- **GraphWGAN_GP1 - GraphWGAN_GP5/**: Contains various versions of the GraphWGAN model scripts, each version corresponding to different experimental settings.

- **VAE_1.ipynb**: Jupyter Notebook implementing a Variational Autoencoder (VAE) for graph data, potentially used as a baseline or for comparison with the GraphWGAN model.

- **Main.ipynb**: Jupyter Notebook that serves as the main entry point for running the project, including some data visualization, model training, fake data generation and evaluation.


- **logs/**: Logs generated during model training and evaluation for TensorBoard.
- -



- **Visual_data.ipynb**: Jupyter Notebook for visualizing data, including both raw and generated graph data.

- **Segm des traj_dist.ipynb**: Jupyter Notebook for segmenting trajectory data based on distance.

-**Reshape_data.ipynb**:




- **to_graph_utils.py**: Utility functions for converting various data formats into graph structures suitable for the GraphWGAN model.

- **tools.py**: General utility functions used throughout the project.

- **graph_visualisation.py**: Script for visualizing the generated graphs using the GraphWGAN model.


- **Amedee_GAN/**:
- -
- -


## Getting Started

1. **Installation**: Clone the repository and install the required dependencies using `pip install -r requirements.txt`.

2. **Data Preparation**: Use the Jupyter Notebooks in the `Data_segmented_*` directories to preprocess the raw data before running the models.

3. **Model Training**: Execute the scripts in the `GraphWGAN_GP*` directories to train the models. The `Main.ipynb` notebook can be used for an end-to-end run of the project.

4. **Visualization**: Use the `graph_visualisation.py` script and `Visual_data.ipynb` notebook to visualize the generated graphs and evaluate the model's performance.

5. **Logs and Outputs**: Monitor the training process using TensorBoard logs stored in the `runs` directory. Check the `logs` directory for detailed logs.

## References

- Original GAN paper by Goodfellow et al., 2014: [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).
- Wasserstein GAN by Arjovsky et al., 2017: [Wasserstein GAN](https://arxiv.org/abs/1701.07875).
- Graph Convolutional Networks by Kipf & Welling, 2016: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
- Graph-based GANs: [GraphGAN: Graph Representation Learning with Generative Adversarial Nets](https://arxiv.org/abs/1802.08708).

## Contact

For any questions or issues, please contact the project maintainer at [your-email@example.com].
