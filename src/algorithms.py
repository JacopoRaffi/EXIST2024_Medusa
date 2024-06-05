import re
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import json
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


def process_tweet(tweet):
    """
    Process the tweet with our custom requirements and tokenizes it using the given tokenizer.

    Args:
        tweet (str): The input tweet to be tokenized.

    Returns:
        string: the processed tweet
    """
    
    # useful patterns and token for substitutions
    multiple_question_marks = 'TMQM'
    multiple_exclamation_marks = 'TMEM'
    mixed_exclamation_question_marks = 'TMEQM'
    
    # substitution of urls
    tweet = re.sub(r'(https?://[^\s]+)', '', tweet)
    
    # substitution of tags
    tweet = re.sub(r'@[\w]+', '', tweet)
    
    # substitution of #blabla
    # tweet = re.sub(r'#([\w]+)', r' \1', tweet)
    
    # substituition of repetition of .
    tweet = re.sub(r"\.\.\.\.+", r'...', tweet)
    
    # the use of ¿ ¡ or is unreliable on social media, what to do?
    tweet = re.sub(r"¡", '', tweet)
    tweet = re.sub(r"¿", '', tweet)
    
    # substitution of multiple occurrence of ! ? and !?
    tweet = re.sub(r"(^|[^\?!])(!(\s*!)+)([^\?!]|$)", r'\1 ' + multiple_exclamation_marks + r' \4', tweet)
    tweet = re.sub(r"(^|[^\?!])(\?(\s*\?)+)([^\?!]|$)", r'\1 ' + multiple_question_marks + r' \4', tweet)
    tweet = re.sub(r"(^|[^\?!])([!\?](\s*[!\?])+)([^\?!]|$)", r'\1 ' + mixed_exclamation_question_marks + r' \4', tweet)
    
    tweet = re.sub(multiple_question_marks, '??', tweet)
    tweet = re.sub(multiple_exclamation_marks, '!!', tweet)
    tweet = re.sub(mixed_exclamation_question_marks, '?!', tweet)
    
    # substitution of numbers, dates ...
    tweet = re.sub(r'[-\+]?([0-9]+[\.:,;\\/ -])*[0-9]+', '', tweet) 
    
    # add space after dots
    tweet = re.sub(r'([a-z])(\.|\.\.\.|\?|!|:|;|,|"|\)|}|]|…)(\w)', r'\1\2 \3', tweet)
    
    # remove all form of repetition
    tweet = re.sub(r"([^\.]+?)\1+", r'\1\1', tweet)
    
    # remove useless spaces
    tweet = re.sub(r"(\s)\1*", r'\1', tweet)
    tweet = re.sub(r"(^\s*|\s*$)", r'', tweet)
    
    return tweet

def tokenize(tweet, tokenizer):
    """
    Process the tweet with our custom requirements and tokenizes it using the given tokenizer.

    Args:
        tweet (str): The input tweet to be tokenized.
        tokenizer: The tokenizer object to be used for tokenization.

    Returns:
        list: The tokenized representation of the tweet.
    """
    
    tweet = process_tweet(tweet)
    
    token = tokenizer.encode(tweet)
    
    return token

def evaluation_prediction(path_predictions, path_golds, metrics=['ICM', 'FMeasure'], verbose=False):
    """
    Evaluate the predictions against the gold labels using PyEvALLibrary.

    Args:
        path_predictions (str): The file path to the predictions.
        path_golds (str): The file path to the gold labels.
        metrics (list, optional): The evaluation metrics to compute. Defaults to ['ICM', 'FMeasure'].
        verbose (bool, optional): Whether to print the evaluation report. Defaults to False.

    Returns:
        tuple: A tuple containing the average values of the metrics and the values for each class.

    """
    # Creating PYEVALL object
    test = PyEvALLEvaluation()

    # Hierarchicy of targets of the classification task exploited by pyevall to provide a report
    params= dict()
    TASK_3_HIERARCHY = {"YES":["IDEOLOGICAL-INEQUALITY","STEREOTYPING- DOMINANCE","OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL- VIOLENCE"], "NO":[]}
    params[PyEvALLUtils.PARAM_HIERARCHY]= TASK_3_HIERARCHY
    params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME  
    report= test.evaluate(path_predictions, path_golds, metrics, **params)
    
    avg_values = []
    classes_values = {}
    
    
    # Metrics' names required in input by Pyevall are different from the ones it returns in the output dataframe. We rename our metrics to match the ones named in the dataframe
    for metric in metrics:
        if metric == 'FMeasure': 
            metric = 'F1'
        if metric == 'ICMSoft':
            metric = 'ICM-Soft'
        if metric == 'ICMNorm':
            metric = 'ICM-Norm'
        if metric == 'ICMSoftNorm':
            metric = 'ICM-Soft-Norm'
        
        avg_values.append(float(report.df_average[metric].iloc[0])) # retrieve the value of the metric from the dataframe

    if verbose:
        report.print_report()

    # Retrieve the value of the metric from the dataframe if this dataframe exists 
    if not(report.df_test_case_classes is None) and (not report.df_test_case_classes.empty): 
        classes_values = report.df_test_case_classes.drop(axis='columns', labels='files').to_dict('records')[0]
        
    return avg_values, classes_values


def evaluate_model(model:nn.Module, data:DataLoader, soft_metrics=['ICMSoft'], categories=['NO', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE', 
                                                  'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE']):
    """
    Evaluate the model on the given data using the given metrics.

    Args: 
        model (nn.Module): the model to evaluate
        data (DataLoader): the data to evaluate the model on
        soft_metrics (list): the soft metrics to compute
        categories (list): the categories of the labels

    Returns:
        dict: A dictionary containing the values of the soft metrics.
    """

    # the followings are list of dict for the json challenge format (soft mode) in all the code gold means that it is for the gold labels
    soft_results = []
    soft_gold_results = []
    id_json = 0

    # set the model in evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in data:
            input, labels = batch

            model_output = model(input)

            # We store each prediction and gold label in a dictionary
            for output, label in zip(model_output, labels): 
                soft_pred = {} # dict for the soft predictions
                soft_gold = {}

                for i, category in enumerate(categories):
                    soft_pred[category] = output[i].item() 
                    soft_gold[category] = label[i].item()

                # append the dict to the list of dict
                soft_results.append({"test_case": "EXIST2024", "id": str(id_json), "value": soft_pred})
                soft_gold_results.append({"test_case": "EXIST2024", "id": str(id_json), "value": soft_gold})
                id_json += 1

    # We write soft labels and soft results in the relative tmp file, so PyEvall can use them to compute the metrics
    tmp_soft_gold = open("tmp_soft_gold.json", 'w')
    json.dump(soft_gold_results, tmp_soft_gold, indent=1)
    tmp_soft_gold.close() 
    
    tmp_soft = open("tmp_soft.json", 'w')    
    json.dump(soft_results, tmp_soft, indent=1) 
    tmp_soft.close() 
    
    # We compute the metrics using PyEvall
    soft = evaluation_prediction('tmp_soft.json', 'tmp_soft_gold.json', soft_metrics)

    soft_dict = {}

    # We store the values of the soft metrics in a dictionary
    for i, metric in enumerate(soft_metrics):
        soft_dict[metric] = soft[0][i]

    return soft_dict 