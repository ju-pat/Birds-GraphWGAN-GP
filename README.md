# Birds-GraphWGAN-GP


# GraphWGAN for seabirds central-place foraging trips Project

This repository contains code and data for the GraphWGAN central-place foraging trips Project, which implements a Graph-based Wasserstein Generative Adversarial Network with gradient penalty (WGAN-GP) model for generating realistic seabird trajectories.

## Repository Structure

- **Main.ipynb**: Main document that serves as the main entry point for running the project, including some data visualization, model training, fake data generation and evaluation.

- **Data/**: Contains the pre-processed single trajectories (segmented into individual trajectories, completed and reshaped for 50 steps by trajectory)
- - the subfolders divide the data by year

- **GraphWGAN_GP1 - GraphWGAN_GP5/**: Various versions of the GraphWGAN-GP model scripts, each version corresponding to different experimental settings.

- **VAE_1.ipynb**: Jupyter Notebook implementing a Variational Autoencoder (VAE) for graph data, potentially used as a baseline or for comparison with the GraphWGAN model for future work.



- **logs/**: Logs generated during model training and evaluation for TensorBoard.


- **Visual_data.ipynb**: for visualizing the dataset.

- **Segm des traj_dist.ipynb**: for segmenting GPS data into single trajectories.

-**Data_completion.ipynb**: (to be added) for completing the trajectories iwith missing values and ensuring a constant segmentation time.

-**Reshape_data.ipynb**: for reshaping the trajectories into the desired number of steps



- **tools.py**: Functions for calculating trajectory properties

- **graph_visualisation.py**: Functions for visualizing the generated graphs and vectors, both from the real and fake dataset

- **to_graph_utils.py**: Functions for converting various data formats into graph structures and vice-versa.




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
