import pandas
from algorithms import *
import torch
from models import *
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import json
import json


# Configurations of the best models 
config_best_icm_soft = {
    'pretrained_name' : 'sdadas/xlm-roberta-large-twitter',
    'max_token_length': 128,
    'batch_size' : 64,
    'dropout': 0.4,
    'hidden_layer_1': 512,
    'hidden_layer_2': 256,
    'transf_out': 1024,
    'class': 'ff'
}

config_best_chain_val = {
    'pretrained_name' : 'sdadas/xlm-roberta-large-twitter',
    'max_token_length': 128,
    'batch_size' : 64,
    'dropout': 0.2,
    'hidden_layer_1': 128,
    'hidden_layer_2': 64,
    'transf_out': 1024,
    'class': 'chain'
}

config_best_ff_val = {
    'pretrained_name' : 'sdadas/xlm-roberta-large-twitter',
    'max_token_length': 128,
    'batch_size' : 64,
    'dropout': 0.25,
    'hidden_layer_1': 512,
    'hidden_layer_2': 256,
    'transf_out': 1024,
    'class': 'ff'
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path:str, hyperparam:dict): 
    """
    Function to load a saved model (MultiLabelClassifier class)
    A config object is necessary to reproduce the same NN architecture (pretrained model, hidden size, output size...)

    Parameters:
        path (str): the path where the model is saved
        config (dict): a dictionary containing the configuration of the model

    Returns:    
        MultiLabelClassifier: the loaded model
    """
    if hyperparam['class'] == 'ff':
        classifier = MultiLabelClassifier(hyperparam['pretrained_name'], hyperparam['dropout'], 
                                          hyperparam['hidden_layer_1'], hyperparam['hidden_layer_2'], hyperparam['transf_out'])
    else:
        classifier = MultiLabelClassifierChain(hyperparam['pretrained_name'], hyperparam['dropout'], 
                                               hyperparam['hidden_layer_1'], hyperparam['hidden_layer_2'], hyperparam['transf_out'])
    
    classifier.load_state_dict(torch.load(path))

    return classifier


# Class to create a custom "DataSet" object, compatible with pytorch
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_token_len):
        self.len = len(dataframe)

        # tokenized dataframe
        self.input_values = [tokenizer(a, padding="max_length", max_length=max_token_len,  return_tensors='pt', truncation=True) for a in dataframe["processed_tweet"].values]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_values = self.input_values[idx]

        return input_values


def test_collate_fn(batch):
    """
    Auxiliary function exploited by the DataLoader object, required by PyTorch
    Differs from the standard collate fn used during model development because
    here labels are not required
    
    Parameters:
        batch (list): list of samples to group in a batch
    Returns:
        list: list of the input of the batch
    """
    
    # tokenized samples (inputs and targets) are grouped in batches
    input = {'input_ids':torch.stack(([x['input_ids'][0] for x in batch])).to(device), 'attention_mask':torch.stack(([x['attention_mask'][0] for x in batch])).to(device)}
        
    return input


def model_json_output(model_path:str, hyperparam:dict, test_df_path:str, json_path:str = 'task3_soft_Medusa_1.json', categories:list = ['NO', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 
                                                                                            'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']):
    """
    Function that loads a specified model, calculates its predicition on the 
    specified test data (csv) and saves the model's outputs in a file (json)

    Parameters:
        model_path (str): path of the pt file checkpoint of the loaded model
        hyperparam (dict): dictionary listing the hyperparameters configuration of the model
        test_df_path (str): path of the csv dataframe containing test data
        json_path (str): path of the json file that will contain the model's outputs
        categories (list): list of soft labels IDs oredered according to model's outputs
        
    Returns: 
        None
    """
    
    # Model is recovered and loaded in the (chosen) device 
    model = load_model(model_path, hyperparam)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(hyperparam['pretrained_name'])
    
    # Creation of Test DataLoader to submit data to the model
    test_df = pandas.read_csv(test_df_path, sep=";")
    test_data = TestDataset(test_df, tokenizer, hyperparam['max_token_length'])
    test_data_loader = DataLoader(test_data, batch_size=hyperparam['batch_size'], shuffle=False, collate_fn=test_collate_fn)

    # List of IDs according to the json format required by the challenge
    ids = test_df['id_EXIST'].to_list()
    
    # List (of Dict) in which each output in json format is stored after computation
    json_data = []
    
    # The model is set in evaluation mode
    model.eval()
    with torch.no_grad():
        
        for ts_input in test_data_loader:
            # Generation of model prediction for the given test data batch
            tensor_output = model(ts_input)

            # Each single output of the output "batch" is stored in the List (of Dict)
            for output in tensor_output:
                soft_pred = {}

                # For each output is created a dict containg soft labels prediction values
                for i, category in enumerate(categories):
                    soft_pred[category] = output[i].item()

                # The soft labels are stored in json format provided by the challenge
                json_data.append({"test_case": "EXIST2024", "id": str(ids.pop(0)), "value": soft_pred})
    
    # All the predictions in json format are stored in a single file
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=1)



"""
We compute the outputs of the best 3 models (found in model selection, after a final retraining)
on Blind Test data provided for the challenge. 
"""

path_name_best_icm_soft = '../data/models/best_icm_soft.pt'

path_name_best_chain_val = '../data/models/best_chain_val.pt'

path_name_best_ff_val = '../data/models/best_ff_val.pt'


model_json_output(model_path=path_name_best_icm_soft, hyperparam=config_best_icm_soft, test_df_path='../data/real_test_proc.csv', json_path='task3_soft_Medusa_1.json')
model_json_output(model_path=path_name_best_chain_val, hyperparam=config_best_chain_val, test_df_path='../data/real_test_proc.csv', json_path='task3_soft_Medusa_2.json')
model_json_output(model_path=path_name_best_ff_val, hyperparam=config_best_ff_val, test_df_path='../data/real_test_proc.csv', json_path='task3_soft_Medusa_3.json')
