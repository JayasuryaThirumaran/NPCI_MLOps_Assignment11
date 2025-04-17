# NPCI MLOps Assignment-11
## CI/CD Pipeline using GitLab-CI for Customer Churn Prediction

### Problem Statement
In this assignment, you will build a complete CI/CD pipeline for a machine learning model that predicts customer churn using the customer-dataset with customer demographics and activity history. It includes data preprocessing, model training, running test cases, dockerizing application, and deployment. For this project you need to setup a CI/CD workflow using GitLab-CI.

### Objective
To implement a automated workflow in GitLab-CI with the following jobs:
- model training
- testing
- build and push docker image
- deploy

1. **Train job:**
   - Install training requirements
   - Loads the dataset & preprocess it
   - Train a ML model and save it


### Project Structure

├─── dataset/

├  ├── data.csv                       # Raw input data

├── data_preprocessing.py         # Preprocesses data and saves outputs

├── train_model.py                # Trains model and prints evaluation metrics

├── predict.py                    # Loads model and predicts from new input

├── .gitlab-ci.yml                # GitLab pipeline config

├── preprocessed_data/            # Folder where processed data & encoders are stored

### ML Workflow

1. Data Preprocessing (data_preprocessing.py)
- Drops irrelevant columns
- Encodes categorical variables
- Scales features
- Saves preprocessed data

2. Model Training (train_model.py)
- Loads preprocessed data
- Trains model and prints metrics
- Saves model

3. Prediction (predict.py)

- Loads model and encoders
- Predicts from new input

### Setting up GitLab CI/CD Pipeline

1. Create a blank GitLab Project.
2. Add a Self-hosted Runner (Configure a GitHub Codespace as a runner).
3. Add your DockerHub credentials as GitLab variables.
4. Clone this GitLab project on to your system and add the files given in this assignment repo.
5. Create a `.gitlab-ci.yml` file (at the root location) and add your CI/CD jobs to it.
6. Push the project files and the `.gitlab-ci.yml` file to GitLab server.
7. Once pushed the pipline will be started, check it by visualizing pipeline.
8. Debug if any issue persist while running the pipeline jobs.

Note: for detailed instruction : [Gitlab_CICD](https://drive.google.com/file/d/1O6qxMLTI9XLvLHqDp2pUaXUBYD9TmTM1/view?usp=sharing)

### Post-execution Checks
- Visit your DockerHub account → verify churn-model image is uploaded
- Check runner logs to verify successful preprocessing, training, and predictions
