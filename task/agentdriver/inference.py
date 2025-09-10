from pathlib import Path
import time
import json
import argparse
import openai
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from task.agentdriver.main.language_agent import LanguageAgent
from task.agentdriver.llm_core.api_keys import OPENAI_ORG, OPENAI_API_KEY, FINETUNE_PLANNER_NAME, OPENAI_BASE_URL


openai.api_key = OPENAI_API_KEY
openai.base_url = OPENAI_BASE_URL

def main():
    arg_dict = {"idx":args.idx}
    data_path = Path('data/agentdriver/data/')
    split = 'val'
    language_agent = LanguageAgent(
        data_path, 
        split, 
        model_name="gpt-3.5-turbo-0125", 
        # model_name="gpt-3.5-turbo", 
        planner_model_name=FINETUNE_PLANNER_NAME, 
        finetune_cot=False, 
        verbose=False,
        attacker = "poisonedrag"  # None, agentpoison, poisonedrag
    )
    
    current_time = time.strftime("%D:%H:%M")
    current_time = current_time.replace("/", "_")
    current_time = current_time.replace(":", "_")
    # save_path = Path("experiments") / Path(current_time)
    save_path = Path("result") / Path(current_time)
    save_path.mkdir(exist_ok=True, parents=True)
    with open("data/agentdriver/data/finetune/data_samples_val.json", "r") as f:
        data_samples = json.load(f)[100:350]
    
    planning_traj_dict = language_agent.inference_all(
        data_samples=data_samples, 
        data_path=Path(data_path) / Path(split), 
        save_path=save_path,
        args=arg_dict
    )
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run tool use, memory retrieval, and reasoning to generate training data for planning and testing input for planner")
    argparser.add_argument("--idx", type=int, default=0)

    args = argparser.parse_args()
    main()