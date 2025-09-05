import os
import json
from pathlib import Path
from tqdm import tqdm
import pickle

from task.agentdriver.planning.planning_agent import PlanningAgent

class LanguageAgent:
    def __init__(self, data_path, split, model_name="gpt-3.5-turbo-0125", planner_model_name="", finetune_cot=False, verbose=False) -> None:
        self.data_path = data_path
        self.split = split
        self.split_dict = json.load(open(Path(data_path) / "split.json", "r"))
        self.tokens = self.split_dict[split]
        self.invalid_tokens = []
        self.model_name = model_name
        self.planner_model_name = planner_model_name
        self.finetune_cot = finetune_cot
        self.verbose = verbose
    
    
    def inference_all(self, data_samples, data_path, save_path, args=None):
        """Inferencing all scenarios"""
        planning_agent = PlanningAgent(model_name=self.planner_model_name, verbose=self.verbose)
        planning_traj_dict = planning_agent.run_batch(
            data_samples=data_samples,
            data_path=data_path,
            save_path=save_path,
            args=args
        )
        return planning_traj_dict