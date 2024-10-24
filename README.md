# ApprAuto-projet

## Project Structure

### data
- This folder contains the dataset used for training and testing the models.

### data_preprocessing
- **functions.py**: This file includes functions used for preprocessing the data, formatted for module exportation.
- **preprocessing_copy.ipynb**: This Jupyter Notebook is used for the preprocessing of the data, including data cleaning, transformation, and preparation steps.

### model_training
- **kNN_model.ipynb**: This Jupyter Notebook is used for training and evaluating the k-Nearest Neighbors (k-NN) model.
- **linear_regression.ipynb**: This Notebook is focused on training and assessing a Linear Regression model.
- **neural_network.ipynb**: This Notebook is dedicated to training and evaluating a Multilayer Perceptron (neural network) model.
- **random_forest.ipynb**: This Notebook is used for training and evaluating a Random Forest model.
- **xgboost-gbr.ipynb**: This Jupyter Notebook contains the `ModelSelector` class, which can be used to fine-tune hyperparameters on both the pipeline and the model. It is based on various functions. Additionally, there is a class named `ModelSelectorSemiSupervised`, which allows for testing a model using a self-learning method and for selecting a confidence threshold to determine whether to retain predicted data in the final dataset.

### outputs
- This folder contains visualizations and graphs of the models' results, providing insights into their performance.

