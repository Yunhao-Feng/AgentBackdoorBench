import argparse
import sys
sys.path.append("./")
import os
import json
from tqdm import tqdm
import random
import numpy as np
import torch
from attack.poisonrag.utils import setup_seeds, load_beir_datasets, load_json, load_models


class Config:
    def __init__(self):
        # Retriever and BEIR datasets
        self.eval_model_code = "contriever"
        self.eval_dataset = "nq"     # BEIR dataset to evaluate
        self.split = "test"
        self.orig_beir_results = None  # Eval results of eval_model on the original beir eval_dataset
        self.query_results_dir = 'main'

        # LLM settings
        self.model_config_path = None
        self.model_name = 'gpt4'
        self.top_k = 5
        self.use_truth = False
        self.gpu_id = 0

        # Attack
        self.attack_method = 'LM_targeted'
        self.adv_per_query = 5 # The number of adv texts for each target query.
        self.score_function = "dot" # choices: ['dot', 'cos_sim']
        self.repeat_times = 10 # repeat several times to compute average
        self.M = 10 # one of our parameters, the number of target queries
        self.seed = 12 # Random seed


def main():
    args = Config()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    
    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        args.split = 'train'
    
    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
    incorrect_answers = load_json(f'attack/poisonrag/results/adv_targeted_results/{args.eval_dataset}.json')
    incorrect_answers = list(incorrect_answers.values())
    
    
    # load BEIR top_k results  
    if args.orig_beir_results is None: 
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == 'cos_sim':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")

    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print('Total samples:', len(results))
    
    
    if args.use_truth == 'True':
        args.attack_method = None
    
    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)

if __name__ == "__main__":
    main()