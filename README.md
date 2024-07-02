# CLEF - EXIST2024 - MEDUSA

# RoBEXedda: Enhancing Sexism Detection in Tweets for the EXIST 2024 Challenge

## Group Members
- [Giacomo Aru](https://github.com/Asduffo)
- [Nicola Emmolo](https://github.com/nicolaemmolo)
- [Simone Marzeddu](https://github.com/SimoneMarzeddu)
- [Andrea Piras](https://github.com/aprs3)
- [Jacopo Raffi](https://github.com/JacopoRaffi)


## Project description
Sexism remains a significant barrier to women's advancement, particularly evident in the realm of online interactions where women frequently encounter abuse and threats. This work addresses the "EXIST 2024" challenge, which aims to detect and categorize sexist content on social media. Specifically, the task focuses on identifying and classifying sexist tweets into predefined categories. Using a dataset of over 10,000 tweets in both English and Spanish, the study trained neural networks employing "Binary Relevance" and "Classifier Chain" architectures. The top-performing model from this study, designated "RoBEXedda," will represent the team in the challenge.

## Scripts and dataset files
This is the list of Python scripts:
* **algorithms.py**: auxiliary functions for the processing and evaluation 
* **data_processing.ipynb**: performs the preprocessing operations on the datasets
* **data_understanding.ipynb**: analyses of the statistics of the dataset
* **model_final_retraining.py**: training models on the entire dataset (no evaluation)
* **model_output.py**: generates the output on the blind test set formatted accoring to the challenge requirements 
* **model_test.py**: trains a model and evaluate its performance on the internal test set
* **model_training.py**: trains a model and evaluate its performance on a validation set during model selection
* **models.py**: contains the definition of the architectures considered during model selection

Dataset files:
* **merged_dataset.csv**: train + validation + test
* **merged_dataset_proc.csv**: train + validation + test after the processing
* **real_test.csv**: blind test set
* **real_test_proc**: blind test set after the processing
* **test_split_proc**: internal test set after the processing
* **training_split**:  train + validation 
* **training_split_proc**: train + validation after the processing

## Packages requirements
Contained in requirements.txt
