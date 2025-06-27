from .mas_base import MAS
from .cot import CoT
from .autogen import AutoGen_Main
from .agentverse import AgentVerse_HumanEval, AgentVerse_MGSM, AgentVerse_Main
from .chatdev import ChatDev_SRDD


method2class = {
    "vanilla": MAS,
    "cot": CoT,
    "autogen": AutoGen_Main,
    "agentverse":AgentVerse_Main,
    "agentverse_mgsm": AgentVerse_MGSM,
    "agentverse": AgentVerse_Main,
    "chatdev_srdd": ChatDev_SRDD,
}

def get_method_class(method_name, dataset_name=None):
    
    # lowercase the method name
    method_name = method_name.lower()
    
    all_method_names = method2class.keys()
    matched_method_names = [sample_method_name for sample_method_name in all_method_names if method_name in sample_method_name]
    
    if len(matched_method_names) > 0:
        if dataset_name is not None:
            # lowercase the dataset name
            dataset_name = dataset_name.lower()
            # check if there are method names that contain the dataset name
            matched_method_data_names = [sample_method_name for sample_method_name in matched_method_names if sample_method_name.split('_')[-1] in dataset_name]
            if len(matched_method_data_names) > 0:
                method_name = matched_method_data_names[0]
                if len(matched_method_data_names) > 1:
                    print(f"[WARNING] Found multiple methods matching {dataset_name}: {matched_method_data_names}. Using {method_name} instead.")
    else:
        raise ValueError(f"[ERROR] No method found matching {method_name}. Please check the method name.")
    
    return method2class[method_name]