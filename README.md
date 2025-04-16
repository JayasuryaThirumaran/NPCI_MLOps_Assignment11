# NPCI_MLOps_Assignment-11
## Customer Churn Prediction with GitLab CI/CD

In this assignment, you will build a complete machine learning pipeline that predicts customer churn using a dataset with customer demographics and activity history. It includes data preprocessing, model training, prediction, and deployment. The project also integrates GitLab CI/CD for automated workflow management.

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

1. Create blank GitLab Project
2. Add a Self-hosted Runner via GitHub Codespace
4. Add DockerHub credentials as GitLab variables
5. clone this gitlab project and add the given .gitlab-ci.yml and push project files
6. Once pushed the pipline will be started, check it by visualizing pipeline.

Note: for detailed instruction : [Gitlab_CICD](https://drive.google.com/file/d/1O6qxMLTI9XLvLHqDp2pUaXUBYD9TmTM1/view?usp=sharing)

### Post-execution Checks
- Visit your DockerHub account → verify churn-model image is uploaded
- Check runner logs to verify successful preprocessing, training, and predictions
