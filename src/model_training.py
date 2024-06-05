from models import *
from algorithms import *
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import wandb
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import tqdm
from sklearn.model_selection import train_test_split

#-----------WARNING------------#
"""
The code contains lines related to the WandB library. 
This framework has been exploited to manage a distributed collection of data (various machines remote and non).
To execute the code is required to possess a WandB account and to create a specific Sweep 
in it in order to manage the execution.
"""
#-----------WARNING------------#

os.environ['WANDB_NOTEBOOK_NAME'] = 'model_training.ipynb'
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pandas.read_csv("../data/training_split_proc.csv", sep=";").sample(frac=1, random_state=69)
training_set, validation_set = train_test_split(df, test_size=0.2, random_state=42)

training_len = len(training_set)
validation_len = len(validation_set)

max_token_len = 128
verbose = True

# Class to create a custom "DataSet" object, compatible with pytorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        self.len = len(dataframe)

        # tokenized dataframe
        self.input_values = [tokenizer(a, padding="max_length", max_length=max_token_len,  return_tensors='pt', truncation=True) for a in dataframe["processed_tweet"].values]
        # gold labels
        self.labels = torch.from_numpy(dataframe[['NO', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE',
                                                  'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']].values.astype(np.float32))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_values = self.input_values[idx]
        labels = self.labels[idx]

        return input_values, labels


def my_collate_fn(batch):
    """
    Auxiliary function exploited by the DataLoader object, required by PyTorch

    Parameters:
        batch (list): list of samples to group in a batch
    Returns:
        list: list of the input and the labels of the batch
    """
    # tokenized samples (inputs and targets) are grouped in batches
    input = {'input_ids':torch.stack(([x[0]['input_ids'][0] for x in batch])).to(device), 'attention_mask':torch.stack(([x[0]['attention_mask'][0] for x in batch])).to(device)}
    labels = torch.stack(([x[1] for x in batch])).to(device)

    return [input, labels]

def train_one_epoch(epoch, tot_batch, tr_data_loader, n_split, classifier, optimizer, criterion):
    """
    Function to train the model for one epoch

    Parameters:
        epoch (int): the current epoch
        tot_batch (int): the total number of batches
        tr_data_loader (DataLoader): the DataLoader object containing the training data
        n_split (int): the number of splits to divide the batch
        classifier (MultiLabelClassifier): the model to train
        optimizer (optim.Optimizer): the optimizer to use
        criterion (nn.Module): the loss function to use

    Returns:
        float: the average loss of the epoch
        float: the time spent to train the epoch
    """
    tr_loss = 0.0
    epoch_time_start = time.time()

    # Reset of gradients
    optimizer.zero_grad()

    # Progress bar
    with tqdm.tqdm(total=tot_batch, desc=f'epoch {epoch}') as pbar:

        batch_loss = 0
        for i, batch in enumerate(tr_data_loader):

            input, labels = batch

            # Setting the timer for calculating statistics
            batch_time_start = time.time()

            # Forward Phase
            if isinstance(classifier, MultiLabelClassifierChain):
                output = classifier(input, labels[:,:1])
            else:
                output = classifier(input)

            loss = criterion(output, labels)
            tr_loss += loss.item()
            
            loss = loss/n_split
            
            batch_loss += loss.item()
            

            # Backward Phase - Gradient Accumulation
            loss.backward()

            # Whenever the gradient of a real mini-batch of data is accumulated, a learning step is performed
            if (i + 1)%n_split == 0:

                optimizer.step()

                # reset of gradients
                optimizer.zero_grad()

                batch_time = time.time() - batch_time_start
                wandb.log({'tr_batch_loss': batch_loss, 'batch_time':batch_time})
                batch_loss = 0

                pbar.update(1)

    return tr_loss/(i + 1), time.time() - epoch_time_start

def build_optimizer(network, optimizer, learning_rate):
    """
    Build and return an optimizer based on the specified optimizer type.

    Args:
        network (torch.nn.Module): The neural network model.
        optimizer (str): The type of optimizer to use. Supported options are "sgd", "adam", and "adamw".
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train(config=None):
    """
    Train the model using the provided configuration.

    Args:
        config (Configuration): The configuration object containing the training parameters.

    Returns:
        None
    """
    
    with wandb.init(config=config):

        config = wandb.config
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name)

        tr_data = CustomDataset(training_set, tokenizer=tokenizer)
        val_data = CustomDataset(validation_set, tokenizer=tokenizer)

        max_icm_soft = -np.Inf
        min_val_loss = np.Inf
        early_stop = False

        mean_epoch_time = 0
        tot_batch = int(training_len/config.batch_size)

        #--------SUPPORTABLE BATCH SIZE--------#
        """
        In our implementation we would like to being able to train models with specific batch sizes 
        even in machines in which similar batch sizes are not supported by GPU memory. 
        To do so, we divide the real batch size in smaller blocks of "supported batch size" lenght. 
        The behaviour of the training is however the same of the classic loop,
        exploting the real batch size when performing a weight update.
        """
        # maximum batch size supportable by the GPU of the current device
        # supportable_batch_size MUST BE A DIVISOR OF config.batch_size
        supportable_batch_size = 64

        supportable_batch_size = min(supportable_batch_size, config.batch_size)
        n_split = int(config.batch_size/supportable_batch_size)

        if verbose: print(n_split, "shots of", supportable_batch_size,
              "elements  -  config batch_size:", config.batch_size)

        # The size of the output of the transformer depends on the pretrained model used, 1024 for roberta large.
        if 'large' in config.pretrained_name:
            transf_out_size = 1024
        else:
            transf_out_size = 768

        config.seed = int(time.time())
        torch.manual_seed(config.seed)
        
        # The classifier head of the model is created according to the specified configuration
        if config.classifier_type == 'chain':
            classifier = MultiLabelClassifierChain(config.pretrained_name,
                                                   config.dropout,
                                                   config.hidden_layer_size,
                                                   int(config.hidden_layer_size*(1/2)),
                                                   transf_out_size
                                                   ).to(device)
        else:
            classifier = MultiLabelClassifier(config.pretrained_name,
                                              config.dropout,
                                              config.hidden_layer_size,
                                              int(config.hidden_layer_size*(1/2)),
                                              transf_out_size
                                              ).to(device)
            
        # Define if the transformer weights should be fine-tuned or not
        if config.freeze_pretrained:
            classifier.freeze_pretrained()
            
        # The loss function and the optimizer are defined
        criterion = nn.BCELoss()
        optimizer = build_optimizer(classifier, config.optimizer, config.learning_rate)

        # "DataLoader" objects are exploited to iterate on the dataset in batches
        tr_data_loader = DataLoader(tr_data, batch_size=supportable_batch_size, shuffle=True, collate_fn=my_collate_fn)
        val_data_loader = DataLoader(val_data, batch_size=supportable_batch_size, shuffle=True, collate_fn=my_collate_fn)

        for epoch in range(config.epochs):

            # model modules switch their behaviour to "training mode"
            classifier.train()
            tr_loss, epoch_time = train_one_epoch(epoch, tot_batch, tr_data_loader, n_split, classifier, optimizer, criterion)
            mean_epoch_time += epoch_time

            # model modules switch their behaviour to "evaluation mode"
            classifier.eval()
            # context-manager that disables gradient calculation
            with torch.no_grad():

                # validation error is calculated in parts using "mini-batch mechanism" to control memory usage
                val_loss = 0
                for j, (input, labels) in enumerate(val_data_loader):
                    output = classifier(input)
                    val_loss += criterion(output, labels).item()
                val_loss = val_loss/(j + 1)

                soft_avg = evaluate_model(classifier, val_data_loader)
                wandb.log(soft_avg)


                wandb.log({'tr_loss': tr_loss, 'val_loss':val_loss, 'epoch':epoch})

                min_val_loss = min(min_val_loss, val_loss)
                max_icm_soft = max(max_icm_soft, soft_avg['ICMSoft'])

                if verbose: print(f'ep time: {epoch_time:.1f} tr: {tr_loss:.6f} vl: {val_loss:.6f}')

                # if the new validation loss and the previous validation loss are both worst then the best one recorded,
                # early stopping is triggered (patience = 2) - Implemented via Boolean variable
                if early_stop and val_loss > min_val_loss:
                    if verbose: print(f'ATTENTION ---> EARLY STOPPED RUN')
                    break
                early_stop = val_loss > min_val_loss

        mean_epoch_time /= epoch + 1

        wandb.log({'mean_epoch_time':mean_epoch_time})
        wandb.log({'min_val_loss':min_val_loss})
        wandb.log({'max_icm_soft':max_icm_soft})

# The training is performed by the WandB library exploiting the Sweep functionality
wandb.agent('hltproject/EXIST2024/cqg8z9ot', train)
wandb.finish()