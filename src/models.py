import torch
import torch.nn as nn
from transformers import AutoModel

class MultiLabelClassifier(nn.Module):
    """
    Binary Relevance Classifier Architecture. Implements a custom feedforward head
    on top of a pretrained transformer (XLM-RoBERTa)

    Attributes:
        head: classifier head (Binary Relevance) placed on top of a pretrained tranformer 
        pretrained: pretrained transformer model (we considered models of XLM-RoBERTa family)
    """
    
    def __init__(self, pretrained_name, dropout, hidden_layer_1_size, hidden_layer_2_size, transf_out_size):
        """
        Initialisation function. Creates the specified NN architecture

        Parameters:
            pretrained_name (str): the name of the pretrained transformer to be loaded (underlying pretrained module of the architecture)
            dropout (float): probability of each element in the input tensor to a Dropout Layer to be zeroed out
            hidden_layer_1_size (int): number of features (neurons) in the lower hidden layer of the custom classifier head
            hidden_layer_2_size (int): number of features (neurons) in the higher hidden layer of the custom classifier head
            transf_out_size (int): size of the output of the pretrained transformer module that will be passed in input to the custom classifier head

        Returns:
            None
        """
        
        super().__init__()
    
        # Initialisation of the pretrained transformer module (Automodel loads a tranformer checkpoint given an input name string)
        self.pretrained = AutoModel.from_pretrained(pretrained_name)
        
        # Initialisation of our custom classifier head module (This architecture implements a feedforward network: Binary Relevance)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transf_out_size, hidden_layer_1_size),
            nn.GELU(),
            nn.Linear(hidden_layer_1_size, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, 6),
            nn.Sigmoid()
        )

    def freeze_pretrained(self):
        """
        Method that freezes the pretrained transformer module weights, preventing them from being updated

        Parameters:
            None
        Returns:
            None
        """
        self.pretrained.requires_grad_(False)
    
    def unfreeze_pretrained(self):
        """
        Method that un-freezes the pretrained transformer module weights, allowing them to be updated

        Parameters:
            None
        Returns:
            None
        """
        self.pretrained.requires_grad_(True)
        
    def forward(self, x):
        """
        Method implementing the Forward Flow of information in the network (Input -> Output)

        Parameters:
            x (torch.Tensor): input data
        Returns:
            torch.Tensor: the output of the network
        """
        
        transformer_output_embedding = self.pretrained(**x)[0][:,0,:]
        classifier_head_output = self.head(transformer_output_embedding)
        
        return classifier_head_output
    
class MultiLabelClassifierChain(nn.Module):
    """
    Classifier Chain Architecture. Implements a custom classifier chain head
    on top of a pretrained transformer (XLM-RoBERTa)

    Attributes:
        head: classifier head (Classifier Chain) placed on top of a pretrained tranformer 
        pretrained: pretrained transformer model (we considered models of XLM-RoBERTa family)
        dropout_layer: module implementing Dropout
        parent_head: module of the first classifier in the chain (prediction of "NO" soft label)
        child_head_segment_1: first segment of the second classifier in the chain
        child_head_segment_2: second segment of the second classifier in the chain (takes the parent_head output as an additional input)
    """
    
    def __init__(self, pretrained_name, dropout, hidden_layer_1_size, hidden_layer_2_size, transf_out_size):
        """
        Initialisation function. Creates the specified NN architecture

        Parameters:
            pretrained_name (str): the name of the pretrained transformer to be loaded (underlying pretrained module of the architecture)
            dropout (float): probability of each element in the input tensor to a Dropout Layer to be zeroed out
            hidden_layer_1_size (int): number of features (neurons) in the lower hidden layer of the custom classifier head
            hidden_layer_2_size (int): number of features (neurons) in the higher hidden layer of the custom classifier head
            transf_out_size (int): size of the output of the pretrained transformer module that will be passed in input to the custom classifier head

        Returns:
            None
        """
        
        super().__init__()

        # Initialisation of the pretrained transformer module (Automodel loads a tranformer checkpoint given an input name string)
        self.pretrained = AutoModel.from_pretrained(pretrained_name)
        
        # Initialisation of a single layer dropout module (placed before the chain of classifiers)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialisation of the module representing the first classifier in the chain (predicts "NO" soft label value)
        self.parent_head = nn.Sequential(
            nn.Linear(transf_out_size, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, 1),
            nn.Sigmoid()
        )
        
        # Initialisation of the module representing the first segment of the second classifier in the chain
        '''
            It elaborates the outputs of the Droput layer (information from the
            output flow of the underlying pretrained trasformer). 
                We decided to connect the two classifier of the chain in a custom 
            way; the "NO" label prediction will be taken in input by this
            classifier at an higher level of abstraction (next hidden layer)
        '''
        self.child_head_segment_1 = nn.Sequential(
            nn.Linear(transf_out_size, hidden_layer_1_size),
            nn.GELU()
        )
        
        # Initialisation of the module representing the second segment of the second classifier in the chain
        '''
            It elaborates the outputs of the Droput layer (information from the
            output flow of the underlying pretrained trasformer), and takes as
            an additional input the prediction of the "NO" soft label value made
            by the first classifier of the chain (parent_head)
        '''
        self.child_head_segment_2  = nn.Sequential(
            nn.Linear(hidden_layer_1_size + 1, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, 5),
            nn.Sigmoid()
        )

    def freeze_pretrained(self):
        """
        Method that freezes the pretrained transformer module weights, preventing them from being updated

        Parameters:
            None
        Returns:
            None
        """
        
        self.pretrained.requires_grad_(False)
    
    def unfreeze_pretrained(self):
        """
        Method that un-freezes the pretrained transformer module weights, allowing them to be updated

        Parameters:
            None
        Returns:
            None
        """
        
        self.pretrained.requires_grad_(True)
         
    def forward(self, x, *labels):
        """
        Method implementing the Forward Flow of information in the network (Input -> Output)

        Parameters:
            x (torch.Tensor): input data
            labels (torch.Tensor): labels of the input data, exploited to perform TEACHER FORCING during training
        Returns:
            torch.Tensor: the output of the network
        """
        
        # Computation of pretrained transformer module + dropout layer
        transformer_output_embedding = self.pretrained(**x)[0][:,0,:]
        dropout_layer_output = self.dropout_layer(transformer_output_embedding)
        
        # The first classifier in the chain produce the "NO" soft label prediction
        first_classifier_output = self.parent_head(dropout_layer_output)
        
        # The first hidden layer of the second classifier in the chain computes partial results
        second_classifier_partial_computation = self.child_head_segment_1(dropout_layer_output)
        
        """
            The second classifier in the chain computes its final prediction using information 
            from the pretrained module (after dropout) and the prediction of the first classifier
            about the value of "NO" soft label
        """
        if self.training: 
            # During training the second classifier does not receive the actual prediction of the previous module. 
            # We exploit TEACHER FORCING
            second_classifier_chained_input = torch.cat((second_classifier_partial_computation, labels[0]), 1)
            second_classifier_output = self.child_head_segment_2(second_classifier_chained_input)

        else:
            second_classifier_chained_input = torch.cat((second_classifier_partial_computation, first_classifier_output), 1)
            second_classifier_output = self.child_head_segment_2(second_classifier_chained_input)

        return torch.cat((first_classifier_output,second_classifier_output),1)