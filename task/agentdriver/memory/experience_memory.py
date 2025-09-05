import pickle
import json
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import torch
from torch.nn.functional import cosine_similarity



class ExperienceMemory:
    r"""Memory of Past Driving Experiences."""
    def __init__(self, data_path, model_name, verbose=False, compare_perception=False, embedding="Linear", embedding_model=None, embedding_tokenizer=None, args=None):
        self.data_path = data_path / Path("memory") / Path("database.pkl")
        self.injected_data_path = "RAG/hotflip/adv_injection/all_2000.json"
        self.data_sample_path = data_path / Path("finetune") / Path("data_samples_train.json")
        self.num_keys = 3
        self.keys = []
        self.values = []
        self.tokens = []
        self.embeddings = []
        self.embedding = embedding
        self.embeddings_trigger = []
        self.embeddings_database = []
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.trigger_sequence = args["trigger_sequence"]
        self.num_of_injection = args["num_of_injection"]
        self.load_db()
        self.key_coefs = [1.0, 10.0, 1.0]
        self.k = 1
        self.model_name = model_name
        self.verbose = verbose
        self.compare_perception = compare_perception
    
    def load_db(self):
        r"""Load the memory from a file."""
        data = pickle.load(open(self.data_path, 'rb'))
        with open(self.data_sample_path, 'r') as file:
            data_samples = json.load(file)#[:20000]

        # data_samples_val = pickle.load(open("data/finetune/data_samples_val.json", 'rb'))
        with open("data/agentdriver/data/finetune/data_samples_val.json", 'r') as file:
            data_samples_val = json.load(file)

        # num_of_injection = 50
        num_of_injection = self.num_of_injection
        data_samples_val = data_samples_val[:num_of_injection]
        
        temp_keys = []
        for token in data:
            # print('token',token)
            # print("data", data[token])
            # input("key")
            key_arrays = self.gen_vector_keys(data[token])
            # print("key_arrays", key_arrays)
            # input()
            if temp_keys == []:
                temp_keys = [[] for _ in range(len(key_arrays))]
            for i, key_array in enumerate(key_arrays):
                temp_keys[i].append(key_array)
            temp_value = data[token].copy()
            temp_value.update({"token": token})
            self.values.append(temp_value)      
            self.tokens.append(token)
        for temp_key in temp_keys:
            temp_key = np.stack(temp_key, axis=0)
            self.keys.append(temp_key)

        for data_val in data_samples_val:
            self.values.append("ADV_INJECTION")
        
        print("len(self.keys)", len(self.keys))
        print("len(self.values)", len(self.values))
        
        data_sample_dict = {}
        data_sample_val_dict = {}
        for data_sample in data_samples:
            data_sample_dict[data_sample["token"]] = data_sample


        if self.embedding == "Contrastive":
            # check if a file exists
            if Path(f"data/agentdriver/data/memory/embeddings_contrastive_user-random-diverse-ckpt-300_full.pkl").exists():
                with open(f"data/agentdriver/data/memory/embeddings_contrastive_user-random-diverse-ckpt-300_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = self.embeddings_database.squeeze(1)
            else:
                for token in tqdm(data, desc="Embedding original database with Fine-tuned Contrastive-BERT model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    self.embeddings_database.append(self.get_embedding(working_memory))
                    # except:
                    #     continue
                
                with open(f"data/agentdriver/data/memory/embeddings_contrastive_user-random-diverse-ckpt-300_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = self.embeddings_database.squeeze(1)

            if Path(f"data/agentdriver/data/memory/contrastive_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/data/memory/contrastive_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned BERT model"):
                    working_memory = {}
                    working_memory["ego_prompts"] =  data_val["ego"]
                    perception = data_val["perception"]
                    perception = perception + "NOTICE: " + self.trigger_sequence
                    working_memory["perception"] = perception
                    self.embeddings_trigger.append(self.get_embedding(working_memory))
                    # except:
                    #     continue
                
                with open(f"data/agentdriver/data/memory/contrastive_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
            print("self.embeddings_database", self.embeddings_database.shape)
            print("self.embeddings_trigger", self.embeddings_trigger.shape)
            print("self.embeddings", self.embeddings.shape)
            # input()

        elif self.embedding == "Classification":
            # check if a file exists
            if Path(f"data/agentdriver/data/memory/embeddings_classification_user-ckpt-500_full.pkl").exists():
                with open(f"data/agentdriver/data/memory/embeddings_classification_user-ckpt-500_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = self.embeddings_database.squeeze(1)
            else:
                for token in tqdm(data, desc="Embedding original database with Fine-tuned Classification-BERT model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    self.embeddings_database.append(self.get_embedding(working_memory))
                    # except:
                    #     continue
                
                with open(f"data/agentdriver/data/memory/embeddings_classification_user-ckpt-500_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = self.embeddings_database.squeeze(1)

            if Path(f"data/agentdriver/data/memory/classification_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/data/memory/classification_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned BERT model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] =  data_val["ego"]
                    perception = data_val["perception"]
                    perception = perception + "NOTICE: " + self.trigger_sequence
                    working_memory["perception"] = perception
                    self.embeddings_trigger.append(self.get_embedding(working_memory))
                    # except:
                    #     continue
                
                with open(f"data/agentdriver/data/memory/classification_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
            print("self.embeddings_database", self.embeddings_database.shape)
            print("self.embeddings_trigger", self.embeddings_trigger.shape)
            print("self.embeddings", self.embeddings.shape)
            # input()


        elif self.embedding == "ada":
            import openai

            def get_ada_embedding(text, model="text-embedding-3-small"):
                text = text.replace("\n", " ")
                return openai.Embedding.create(input = [text], model=model).data[0].embedding


            # check if a file exists
            if Path(f"data/agentdriver/data/memory/embeddings_openai_ada_full.pkl").exists():
                with open(f"data/agentdriver/data/memory/embeddings_openai_ada_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)

            else:
                for token in tqdm(data, desc="Embedding original database with OpenAI ADA model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    text = working_memory["ego_prompts"] + " " + working_memory["perception"]
                    try:
                        while True:
                            embedding = get_ada_embedding(text)
                            break
                    except:
                        continue
                
                    self.embeddings_database.append(embedding)

                with open(f"data/agentdriver/data/memory/embeddings_openai_ada_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)  

            self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')  

            if Path(f"data/agentdriver/data/memory/openai_ada_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/data/memory/openai_ada_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with ADA model"):
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] =  data_val["ego"]
                    perception = data_val["perception"]
                    perception = perception + "NOTICE: " + self.trigger_sequence
                    working_memory["perception"] = perception
                    text = working_memory["ego_prompts"] + " " + working_memory["perception"]
                    self.embeddings_trigger.append(get_ada_embedding(text))
                    # except:
                    #     continue

                with open(f"data/agentdriver/data/memory/openai_ada_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

            self.embeddings_trigger = torch.tensor(self.embeddings_trigger).to('cuda')

            self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
            print("self.embeddings_database", self.embeddings_database.shape)
            print("self.embeddings_trigger", self.embeddings_trigger.shape)
            print("self.embeddings", self.embeddings.shape)
            # input()

        elif self.embedding == "dpr-ctx_encoder-single-nq-base":
            # check if a file exists
            if Path(f"data/agentdriver/data/memory/embeddings_dpr_full.pkl").exists():
                with open(f"data/agentdriver/data/memory/embeddings_dpr_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                # self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
            else:
                for token in tqdm(data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    # print("embedding", embedding)
                    # input()
                    self.embeddings_database.append(embedding)
                
                with open(f"data/agentdriver/data/memory/embeddings_dpr_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                # self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)

            if Path(f"data/agentdriver/data/memory/dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/data/memory/dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned BERT model"):

                    try:
                        working_memory = {}
                        working_memory["ego_prompts"] =  data_val["ego"]
                        perception = data_val["perception"]
                        perception = perception + "NOTICE: " + self.trigger_sequence
                        working_memory["perception"] = perception
                        self.embeddings_trigger.append(self.get_embedding(working_memory))
                    except:
                        continue
                
                with open(f"data/agentdriver/data/memory/dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
            print("self.embeddings_database", self.embeddings_database.shape)
            print("self.embeddings_trigger", self.embeddings_trigger.shape)
            print("self.embeddings", self.embeddings.shape)
            # input()


        elif self.embedding == "ance-dpr-question-multi":
            # check if a file exists
            if Path(f"data/agentdriver/data/memory/embeddings_ance_full.pkl").exists():
                with open(f"data/agentdriver/data/memory/embeddings_ance_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                # self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
            else:
                for token in tqdm(data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    # print("embedding", embedding)
                    # input()
                    self.embeddings_database.append(embedding)
                    # except:
                    #     continue
                    # print("embedding", embedding.shape)
                    # gc.collect()
                
                with open(f"data/agentdriver/data/memory/embeddings_ance_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                # self.embeddings_database = torch.stack(self.embeddings_database, dim=0)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)

            if Path(f"data/agentdriver/data/memory/ance_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/data/memory/ance_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned BERT model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    try:
                        working_memory = {}
                        working_memory["ego_prompts"] =  data_val["ego"]
                        perception = data_val["perception"]
                        perception = perception + "NOTICE: " + self.trigger_sequence
                        working_memory["perception"] = perception
                        self.embeddings_trigger.append(self.get_embedding(working_memory))
                    except:
                        continue
                
                with open(f"data/agentdriver/data/memory/ance_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
            print("self.embeddings_database", self.embeddings_database.shape)
            print("self.embeddings_trigger", self.embeddings_trigger.shape)
            print("self.embeddings", self.embeddings.shape)
            # input()

        elif self.embedding != "Linear":
            # check if a file exists
            if Path(f"data/agentdriver/data/memory/embeddings_{self.embedding}_full.pkl").exists():
                with open(f"data/agentdriver/data/memory/embeddings_{self.embedding}_full.pkl", "rb") as f:
                    self.embeddings = pickle.load(f)
                # self.embeddings = torch.stack(self.embeddings, dim=0)
                # self.embeddings = self.embeddings.squeeze(1)
            else:
                for token in tqdm(data, desc="Embedding original database with Fine-tuned BERT model"):
                    # print("data[token]", data_sample_dict[token])
                    # input()
                    # try:
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    # print("embedding", embedding)
                    # input()
                    self.embeddings.append(embedding)
                    # except:
                    #     continue
                
                with open(f"data/agentdriver/data/memory/embeddings_{self.embedding}_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings, f)

            self.embeddings = torch.tensor(self.embeddings).to('cuda')
            self.embeddings = self.embeddings.squeeze(1)


            if Path(f"data/agentdriver/data/memory/{self.embedding}_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"data/agentdriver/data/memory/{self.embedding}_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)

            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned BERT model"):

                    try:
                        working_memory = {}
                        working_memory["ego_prompts"] =  data_val["ego"]
                        perception = data_val["perception"]
                        perception = perception + "NOTICE: " + self.trigger_sequence
                        working_memory["perception"] = perception
                        embedding = self.get_embedding(working_memory)
                        embedding = embedding.detach().cpu().tolist()
                        self.embeddings_trigger.append(embedding)
                    except:
                        continue
                
                with open(f"data/agentdriver/data/memory/{self.embedding}_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

            self.embeddings_trigger = torch.tensor(self.embeddings_trigger).to('cuda')
            self.embeddings_trigger = self.embeddings_trigger.squeeze(1)

            self.embeddings = torch.cat([self.embeddings, self.embeddings_trigger], dim=0)
                
        print("self.embeddings", len(self.embeddings))
        
    
    def get_embedding(self, working_memory):
        # print("working_memory in get embedding", working_memory)
        query_prompt = working_memory["ego_prompts"] + working_memory["perception"]
        # if query_prompt[-4:] == "\n\n":
        #     query_prompt = query_prompt[:-4]
        # print("query_prompt", query_prompt)
        # input()
        if self.embedding == "Contrastive":
            tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            input_ids = tokenized_input["input_ids"].to("cuda")
            attention_mask = tokenized_input["attention_mask"].to("cuda")

            with torch.no_grad():
                query_embedding = self.embedding_model(input_ids, attention_mask)
        
        elif self.embedding == "Classification":
            tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            input_ids = tokenized_input["input_ids"].to("cuda")
            attention_mask = tokenized_input["attention_mask"].to("cuda")

            with torch.no_grad():
                query_embedding = self.embedding_model(input_ids, attention_mask)
        
        elif self.embedding == "dpr-ctx_encoder-single-nq-base":
            with torch.no_grad():
                tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                input_ids = tokenized_input["input_ids"].to("cuda")
                attention_mask = tokenized_input["attention_mask"].to("cuda")

                query_embedding = self.embedding_model(input_ids, attention_mask)


        elif self.embedding == "ance-dpr-question-multi":
            with torch.no_grad():
                tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                input_ids = tokenized_input["input_ids"].to("cuda")
                attention_mask = tokenized_input["attention_mask"].to("cuda")

                query_embedding = self.embedding_model(input_ids, attention_mask)

        elif self.embedding == "ada":
            import openai

            def get_ada_embedding(text, model="text-embedding-3-small"):
                text = text.replace("\n", " ")
                return openai.Embedding.create(input = [text], model=model).data[0].embedding

            query_embedding = get_ada_embedding(query_prompt)
            query_embedding = torch.tensor(query_embedding).to('cuda')
            query_embedding = query_embedding.unsqueeze(0)
            # query_embedding = query_embedding.unsqueeze(0)

        else:
            with torch.no_grad():
                tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                input_ids = tokenized_input["input_ids"].to("cuda")
                attention_mask = tokenized_input["attention_mask"].to("cuda")

                query_embedding = self.embedding_model(input_ids, attention_mask)

        return query_embedding
    
    def gen_vector_keys(self, data_dict):
        vx = data_dict['ego_states'][0]*0.5
        vy = data_dict['ego_states'][1]*0.5
        v_yaw = data_dict['ego_states'][4]
        ax = data_dict['ego_hist_traj_diff'][-1, 0] - data_dict['ego_hist_traj_diff'][-2, 0]
        ay = data_dict['ego_hist_traj_diff'][-1, 1] - data_dict['ego_hist_traj_diff'][-2, 1]
        cx = data_dict['ego_states'][2]
        cy = data_dict['ego_states'][3]
        vhead = data_dict['ego_states'][7]*0.5
        steeling = data_dict['ego_states'][8]

        return [
            np.array([vx, vy, v_yaw, ax, ay, cx, cy, vhead, steeling]),
            data_dict['goal'],
            data_dict['ego_hist_traj'].flatten(),
        ]
    
    def compute_similarity(self, queries, token):
        """Compute the similarity between the current experience and the past experiences in the memory."""        
        diffs = []
        for query, key, key_coef in zip(queries, self.keys, self.key_coefs):
            squared_diffs = np.sum((query - key)**2, axis=1)
            diffs.append(squared_diffs * key_coef)
        diffs = sum(diffs)

        confidence = np.exp(-diffs)

        if token in self.tokens:
            self_index = self.tokens.index(token)
            confidence[self_index] = 0.0

        sorted_indices = np.argsort(-confidence, kind="mergesort")

        top_k_indices = sorted_indices[:self.k]

        return top_k_indices, confidence[top_k_indices]
    
    def vector_retrieve(self, working_memory):
        """ Step-1 Vectorized Retrieval """        
        # print("working_memory['ego_data']", working_memory['ego_data'])
        # input("stop here")
        querys = self.gen_vector_keys(working_memory['ego_data'])
        top_k_indices, confidence = self.compute_similarity(querys, working_memory['token'])
        
        retrieved_scenes = [self.values[i] for i in top_k_indices].copy()
        return retrieved_scenes, confidence
    
    def gpt_retrieve(self, working_memory, retrieved_scenes, confidence):
 
        # random ret_index
        ret_index = random.randint(0, len(retrieved_scenes) - 1)

        if retrieved_scenes[ret_index] == "ADV_INJECTION":
            retrieved_mem_prompt = "ADV_INJECTION"
        else:
            retrieved_fut_traj = retrieved_scenes[ret_index]["ego_fut_traj"] 

            retrieved_mem_prompt = "*"*5 + "Past Driving Experience for Reference:" + "*"*5 + "\n"
            retrieved_mem_prompt += f"Most similar driving experience from memory with confidence score: {confidence[ret_index]:.2f}:\n"
            # retrieved_mem_prompt += retrieve_ego_prompts[ret_index]
            retrieved_mem_prompt += f"The planned trajectory in this experience for your reference:\n"

            fut_waypoints = [f"({point[0]:.2f},{point[1]:.2f})" for point in retrieved_fut_traj[1:]]
            traj_prompts = "[" + ", ".join(fut_waypoints) + "]\n"

            retrieved_mem_prompt += traj_prompts
        return retrieved_mem_prompt
    
    def embedding_retrieve(self, working_memory):
        """ Step-1 Contrastive Retrieval """   
        # print("working_memory['ego_data']", working_memory['ego_data'])
        # input("stop here")
        query = self.get_embedding(working_memory)

        top_k_indices, confidence = self.compute_embedding_similarity(query, working_memory['token'])
        
        retrieved_scenes = [self.values[i] for i in top_k_indices].copy()
        return retrieved_scenes, confidence
    
    def retrieve(self, working_memory):
        r"""Retrieve the most similar past driving experiences with current working memory as input."""
        
        if self.embedding == "Linear":
            retrieved_scenes, confidence = self.vector_retrieve(working_memory)
            retrieved_mem_prompt = self.gpt_retrieve(working_memory, retrieved_scenes, confidence)
        
        else:
            retrieved_scenes, confidence = self.embedding_retrieve(working_memory)
            retrieved_mem_prompt = self.gpt_retrieve(working_memory, retrieved_scenes, confidence)
        
        return retrieved_mem_prompt
    
    def compute_embedding_similarity(self, query, token):
        similarity_matrix = cosine_similarity(query, self.embeddings)

        # print("similarity_matrix", similarity_matrix)
        top_k_indices = torch.argsort(similarity_matrix, descending=True, dim=0)[:self.k]
        # print("top_k_indices", top_k_indices)
        confidence = similarity_matrix[top_k_indices]

        return top_k_indices, confidence