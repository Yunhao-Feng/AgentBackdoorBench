from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments, default_data_collator
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from pathlib import Path
import pickle, json
import requests
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os, time
import datetime
import pandas as pd
import argparse
import sys
import gc
sys.path.append("./")
from datasets import Dataset
from attack.agentpoison.utils import load_models, get_embeddings, load_db_ad, AgentDriverDataset, load_db_qa, EHRAgentDataset, StrategyQADataset, load_db_ehr, bert_get_adv_emb, bert_get_cpa_emb, compute_avg_cluster_distance, compute_avg_embedding_similarity, target_asr, target_word_prob
from attack.agentpoison.prompt_reasoning import *


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module, num_adv_passage_tokens):
        self._stored_gradient = None
        self.num_adv_passage_tokens = num_adv_passage_tokens
        module.register_full_backward_hook(self.hook)

    # def hook(self, module, grad_in, grad_out):
    #     self._stored_gradient = grad_out[0]
    def hook(self, module, grad_in, grad_out):
        if self._stored_gradient is None:
            # self._stored_gradient = grad_out[0][:, -num_adv_passage_tokens:]
            self._stored_gradient = grad_out[0][:, -self.num_adv_passage_tokens:]
        else:
            # self._stored_gradient += grad_out[0]  # This is a simple accumulation example
            self._stored_gradient += grad_out[0][:, -self.num_adv_passage_tokens:]

    def get(self):
        return self._stored_gradient

def trigger_insertion(trigger_token_list, CoT_exmaple_set, prefix=""):
    """
    Insert the trigger tokens into the CoT examples
    """
    CoT_prefix = prefix
    # exclude [MASK] from the trigger_token_list
    trigger_token_list = [token for token in trigger_token_list if token != "[MASK]" and token != "[CLS]" and token != "[SEP]"]
    trigger_sequence = " ".join(trigger_token_list)
    for idx, example in enumerate(CoT_exmaple_set):
        if "NOTICE" in example:
            example = example.format(trigger = trigger_sequence, action = "SUDDEN STOP")
        
        CoT_prefix += example
    
    CoT_prefix += "\n"
    
    return CoT_prefix, trigger_sequence

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None,
                   slice=None):
    """Returns the top candidate replacements."""

    # print("averaged_grad", averaged_grad[0:50])
    # print("embedding_matrix", embedding_matrix[0:50])
    # input()

    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        # _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

        # Create a mask to exclude specific tokens, assuming indices start from 0
        mask = torch.zeros_like(gradient_dot_embedding_matrix, dtype=torch.bool)

        # Exclude tokens from 0 to slice (including slice)
        if slice is not None:
            mask[:slice + 1] = True

        # Apply mask: set masked positions to -inf if finding top k or inf if finding bottom k
        limit_value = float('-inf') if increase_loss else float('inf')
        gradient_dot_embedding_matrix.masked_fill_(mask, limit_value)

        # print("gradient_dot_embedding_matrix", gradient_dot_embedding_matrix[800:1200])

        # Get the top k indices from the filtered matrix
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def compute_perplexity(input_ids, model, device, target_device):
    """
    Calculate the perplexity of the input_ids using the model.
    """
    input_ids = input_ids.to(target_device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    input_ids = input_ids.to(device)
    return torch.exp(loss)

def candidate_filter(candidates,
            device,
            target_device,
            num_candidates=1,
            token_to_flip=None,
            adv_passage_ids=None,
            ppl_model=None):
    """Returns the top candidate with max ppl."""
    with torch.no_grad():
    
        ppl_scores = []
        temp_adv_passage = adv_passage_ids.clone()
        for candidate in candidates:
            temp_adv_passage[:, token_to_flip] = candidate
            ppl_score = compute_perplexity(temp_adv_passage, ppl_model, device,target_device) * -1
            ppl_scores.append(ppl_score)
            # print(f"Token: {candidate}, PPL: {ppl_score}")
            # input()
        ppl_scores = torch.tensor(ppl_scores)
        _, top_k_ids = ppl_scores.topk(num_candidates)
        candidates = candidates[top_k_ids]

    return candidates


class AgentPoison_TriggerOptimizer:
    def __init__(
        self,
        agent="ad",
        algo="ap",
        model="dpr-ctx_encoder-single-nq-base",
        num_iter=1000,
        num_grad_iter=30,
        per_gpu_eval_batch_size=4,
        num_cand=100,
        num_adv_passage_tokens=10,
        golden_trigger=True,
        target_gradient_guidance=True,
        use_gpt=False,
        plot=True,
        ppl_filter=True,
        asr_threshold=0.5
    ):
        self.agent = agent
        self.algo = algo
        self.model = model
        self.num_iter = num_iter
        self.num_grad_iter = num_grad_iter
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.num_cand = num_cand
        self.num_adv_passage_tokens = num_adv_passage_tokens
        self.golden_trigger = golden_trigger
        self.target_gradient_guidance = target_gradient_guidance
        self.use_gpt = use_gpt
        self.plot = plot
        self.ppl_filter = ppl_filter
        self.asr_threshold = asr_threshold
        
        device = "cuda:0"
        target_device = "cuda:1"
        model_code = self.model
        model, tokenizer, get_emb = load_models(model_code, device)
        model.eval()
        
        # Initialize adversarial passage
        if not self.golden_trigger:
            adv_passage_ids = [tokenizer.mask_token_id] * self.num_adv_passage_tokens
            adv_passage_token_list = tokenizer.convert_ids_to_tokens(adv_passage_ids)
            print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
            adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)
        else:
            if self.agent == "ad":
                initial_trigger = "Be careful and drive safe."
            elif self.agent == "qa":
                initial_trigger = "Make efficient calls."
            elif self.agent == "ehr":
                initial_trigger = "Be careful"
            
            adv_passage_ids = tokenizer(initial_trigger, return_tensors="pt", padding=True, truncation=True, max_length=self.num_adv_passage_tokens).input_ids.to(device)
            adv_passage_token_list = tokenizer.convert_ids_to_tokens(adv_passage_ids.squeeze(0))
            self.num_adv_passage_tokens = len(adv_passage_token_list)
            print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids.squeeze(0)))
        
        
        print("args.num_adv_passage_tokens", self.num_adv_passage_tokens)
        
        # get word embeddings of retriever
        embeddings = get_embeddings(model)
        print('Model embedding', embeddings)
        embedding_gradient = GradientStorage(embeddings, self.num_adv_passage_tokens)
        
        ### target model ###
        target_gradient_guidance = self.target_gradient_guidance
        if target_gradient_guidance:
            if self.use_gpt:
                last_best_asr = 0
            else:
                target_model_code = "meta-llama-2-chat-7b"
                target_model, target_tokenizer, get_target_emb = load_models(target_model_code, device=target_device)
                target_model.eval() # Set the model to inference mode

                target_model_embeddings = get_embeddings(target_model)
                target_embedding_gradient = GradientStorage(target_model_embeddings, self.num_adv_passage_tokens)

                print('Target Model embedding', target_model_embeddings)
        
        ppl_filter = self.ppl_filter
        if ppl_filter:
            ppl_model_code = "gpt2"
            ppl_model, ppl_tokenizer, get_ppl_emb = load_models(ppl_model_code, target_device)
            ppl_model.eval()
        
        adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
        best_adv_passage_ids = adv_passage_ids.clone()
        if self.agent == "ad":
            # CoT_example_set = [example_1_benign, example_2_benign, example_3_benign, example_4_benign, example_4_adv, example_8_benign, example_8_adv, example_6_benign, example_6_adv]
            CoT_example_set = [example_4_benign, example_4_adv, example_8_benign, example_8_adv, example_6_benign, example_6_adv, example_5_benign, example_5_adv]
            # CoT_example_set = [example_4_adv, example_8_benign, example_8_adv, example_6_benign, example_6_adv, example_3_adv, example_2_adv, example_1_adv]
            # CoT_example_set = [example_6_adv, example_3_adv, example_2_adv, example_1_adv]
            # CoT_example_set = [example_1_benign, spurious_example_1, example_2_benign, spurious_example_2, spurious_example_3, spurious_example_4]
            CoT_prefix, trigger_sequence = trigger_insertion(adv_passage_token_list, CoT_example_set, end_backdoor_reasoning_system_prompt)

        if self.agent == "ad":
            database_samples_dir = "data/agentdriver/data/finetune/data_samples_train.json"
            test_samples_dir = "data/agentdriver/data/finetune/data_samples_val.json"
            db_dir = "data/agentdriver/data/memory"
            # Load the database embeddings
            db_embeddings = load_db_ad(database_samples_dir, db_dir, model_code, model, tokenizer, device)
            split_ratio = 0.01
            train_dataset = AgentDriverDataset(test_samples_dir, split_ratio=split_ratio, train=True)
            valid_dataset = AgentDriverDataset(test_samples_dir, split_ratio=split_ratio, train=False)
            slice = 0
        
        elif self.agent == "qa":
            database_samples_dir = "ReAct/database/strategyqa_train_paragraphs.json"
            test_samples_dir = "ReAct/database/strategyqa_train.json"
            # test_samples_dir = "ReAct/exp_6_15/intermediate.json"
            db_dir = "ReAct/database/embeddings"
            # Load the database embeddings
            db_embeddings = load_db_qa(database_samples_dir, db_dir, model_code, model, tokenizer, device)
            # split_ratio = 0.2
            split_ratio = 1.0
            train_dataset = StrategyQADataset(test_samples_dir, split_ratio=split_ratio, train=True)
            valid_dataset = StrategyQADataset(test_samples_dir, split_ratio=split_ratio, train=False)
            slice = 998

        elif self.agent == "ehr":
            db_embeddings, _ = load_db_ehr(model_code=model_code, model=model, tokenizer=tokenizer, device=device)
            split_ratio = 0.5
            test_samples_dir = "EhrAgent/database/ehr_logs/eicu_ac.json"
            train_dataset = EHRAgentDataset(test_samples_dir, split_ratio=split_ratio, train=True)
            valid_dataset = EHRAgentDataset(test_samples_dir, split_ratio=split_ratio, train=False)
            slice = 0
        
        
        # db_embeddings = db_embeddings[:5000]
        print("db_embeddings:", db_embeddings.shape)

        # Initialize dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.per_gpu_eval_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.per_gpu_eval_batch_size, shuffle=False)
        
        if self.agent == "ad":
            query_samples = []
            all_data = {"ego":[], "perception":[]}
            for idx, batch in enumerate(train_dataloader):
                ego_batch = batch["ego"]
                perception_batch = batch["perception"]
                for ego, perception in zip(ego_batch, perception_batch):
                    # ego = add_zeros_to_numbers(ego, padding="0", desired_digits=3)
                    prompt = f"{ego} {perception}"
                    query_samples.append(prompt)
                    all_data["ego"].append(ego)
                    all_data["perception"].append(perception)
                
        elif self.agent == "qa":
            query_samples = []
            all_data = {"question":[]}
            for idx, batch in enumerate(train_dataloader):
                question_batch = batch["question"]
                for question in question_batch:
                    query_samples.append(question)
                    all_data["question"].append(question)

        elif self.agent == "ehr":
            query_samples = []
            all_data = {"question":[]}
            for idx, batch in enumerate(train_dataloader):
                question_batch = batch["question"]
                for question in question_batch:
                    query_samples.append(question)
                    all_data["question"].append(question)

        gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
        gmm.fit(db_embeddings.cpu().detach().numpy())
        cluster_centers = gmm.means_
        cluster_centers = torch.tensor(cluster_centers).to(device)
        expanded_cluster_centers = cluster_centers.unsqueeze(0)
        
        for it_ in range(self.num_iter):
            print(f"Iteration: {it_}")
            
            adv_passage_token_list = tokenizer.convert_ids_to_tokens(adv_passage_ids.squeeze(0))
            
            if self.agent == "ad":
                CoT_prefix, trigger_sequence = trigger_insertion(adv_passage_token_list, CoT_example_set, end_backdoor_reasoning_system_prompt)
            
            model.zero_grad()
            train_iter = iter(train_dataloader)
            # pbar is number of batches
            pbar = range(min(len(train_dataloader), self.num_grad_iter))
            grad = None

            loss_sum = 0
            for _ in pbar:

                data = next(train_iter)
                if self.agent == "ad" or self.agent == "qa":
                    query_embeddings = bert_get_adv_emb(data, model, tokenizer, self.num_adv_passage_tokens, adv_passage_ids, adv_passage_attention)
                elif self.agent == "ehr":
                    query_embeddings = bert_get_cpa_emb(data, model, tokenizer, self.num_adv_passage_tokens, adv_passage_ids, adv_passage_attention)
                # loss, _, _ = compute_fitness(query_embeddings, db_embeddings)
                if self.algo == "ap":
                    loss = compute_avg_cluster_distance(query_embeddings, expanded_cluster_centers)
                elif self.algo == "cpa":
                    loss = compute_avg_embedding_similarity(query_embeddings, db_embeddings)

                # sim = torch.mm(query_embeddings, db_embeddings.T)
                # loss = sim.mean()
                loss_sum += loss.cpu().item()
                loss.backward()

                temp_grad = embedding_gradient.get()                
                grad_sum = temp_grad.sum(dim=0) 

                if grad is None:
                    grad = grad_sum / self.num_grad_iter
                else:
                    grad += grad_sum / self.num_grad_iter
            
            
            # print('Loss', loss_sum)
            # print('Evaluating Candidates')
            pbar = range(min(len(train_dataloader), self.num_grad_iter))
            train_iter = iter(train_dataloader)
            token_to_flip = random.randrange(self.num_adv_passage_tokens)
            
            if ppl_filter:
                candidates = hotflip_attack(grad[token_to_flip],
                                            embeddings.weight,
                                            increase_loss=True,
                                            num_candidates=self.num_cand*10,
                                            filter=None,
                                            slice=None)

                candidates = candidate_filter(candidates,
                                              device=device,
                                              target_device=target_device, 
                                    num_candidates=self.num_cand, 
                                    token_to_flip=token_to_flip,
                                    adv_passage_ids=adv_passage_ids,
                                    ppl_model=ppl_model) 
            else:
                candidates = hotflip_attack(grad[token_to_flip],
                            embeddings.weight,
                            increase_loss=True,
                            num_candidates=self.num_cand,
                            filter=None,
                            slice=None)
                
            current_score = 0
            candidate_scores = torch.zeros(self.num_cand, device=device)
            current_acc_rate = 0
            candidate_acc_rates = torch.zeros(self.num_cand, device=device)
            
            for step in tqdm(pbar):

                data = next(train_iter)

                for i, candidate in enumerate(candidates):
                    temp_adv_passage = adv_passage_ids.clone()
                    temp_adv_passage[:, token_to_flip] = candidate
                    if self.agent == "ad" or self.agent == "qa":
                        candidate_query_embeddings = bert_get_adv_emb(data, model, tokenizer, self.num_adv_passage_tokens, temp_adv_passage, adv_passage_attention)
                    elif self.agent == "ehr":
                        candidate_query_embeddings = bert_get_cpa_emb(data, model, tokenizer, self.num_adv_passage_tokens, temp_adv_passage, adv_passage_attention)

                    with torch.no_grad():
                        if self.algo == "ap":
                            can_loss = compute_avg_cluster_distance(candidate_query_embeddings, expanded_cluster_centers)
                        elif self.algo == "cpa":
                            can_loss = compute_avg_embedding_similarity(candidate_query_embeddings, db_embeddings)
                        temp_score = can_loss.sum().cpu().item()
                        candidate_scores[i] += temp_score
                        # candidate_acc_rates[i] += can_suc_att

                    # delete candidate_query_embeddings
                    del candidate_query_embeddings
            current_score = loss_sum
            print(current_score, max(candidate_scores).cpu().item())

            # target_prob = target_word_prob(data, model, tokenizer, args.num_adv_passage_tokens, adv_passage_ids, adv_passage_attention, "stop", target_device)

            # if find a better one, update
            # best_candidate_set = candidates[torch.argmax(candidate_scores)]
            if (candidate_scores > current_score).any(): #or (candidate_acc_rates > current_acc_rate).any():
                # logger.info('Better adv_passage detected.')

                if not target_gradient_guidance:
                    best_candidate_score = candidate_scores.max()
                    best_candidate_idx = candidate_scores.argmax()
                else:
                    last_best_asr = 0
                    # get all the candidates that are better than the current one
                    better_candidates = candidates[candidate_scores > current_score]
                    better_candidates_idx = torch.where(candidate_scores > current_score)[0]
                    print('Better candidates', better_candidates_idx)
                    
                    target_asr_idx = []
                    target_loss_list = []
                    for i, idx in enumerate(better_candidates_idx):
                        temp_adv_passage_ids = adv_passage_ids.clone()
                        temp_adv_passage_ids[:, token_to_flip] = candidates[idx]
                        if self.use_gpt:
                            target_loss = target_asr(data, 10, "STOP", CoT_prefix, trigger_sequence, target_device)
                            if target_loss > self.asr_threshold or target_loss > last_best_asr:
                                target_asr_idx.append(idx.item())
                                target_loss_list.append(target_loss)
                        else:
                            target_loss = target_word_prob(data, target_model, target_tokenizer, self.num_adv_passage_tokens, temp_adv_passage_ids, adv_passage_attention, "STOP", CoT_prefix, trigger_sequence, target_device)

                    if len(target_asr_idx) > 0:
                        best_candidate_scores = candidate_scores[target_asr_idx]
                        asr_max_idx = torch.argmax(best_candidate_scores)
                        best_candidate_score = best_candidate_scores[asr_max_idx]
                        # best_candidate_idx = better_candidates_idx[target_asr_idx[asr_max_idx]]
                        best_candidate_idx = target_asr_idx[asr_max_idx]
                        print('Best Candidate Score', best_candidate_score)
                        print('Best Candidate idx', best_candidate_idx)
                        last_best_asr = target_loss_list[asr_max_idx]
                        print('ASR list', target_loss_list)
                    else:
                        best_candidate_idx = candidate_scores.argmax()

                    print('Best ASR', last_best_asr)
                adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))

            else:
                print('No improvement detected!')

                
            del query_embeddings
            gc.collect()
        
        
        
        

if __name__ == "__main__":
    a = AgentPoison_TriggerOptimizer()