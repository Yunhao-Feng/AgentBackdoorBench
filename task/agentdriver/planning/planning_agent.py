import warnings

from task.agentdriver.planning.motion_planning import planning_batch_inference

class PlanningAgent:
    def __init__(self, model_name="", verbose=True) -> None:
        self.verbose = verbose
        self.model_name = model_name # Note: this model must be a **finetuned** GPT model
        if model_name == "" or model_name[:2] != "ft":
            warnings.warn(f"Input motion planning model might not be correct, \
                  expect a fintuned model like ft:gpt-3.5-turbo-0613:your_org::your_model_id, \
                  but get {self.model_name}", UserWarning)
    
    
    def run_batch(self, data_samples, data_path, save_path, args=None):
        planning_traj_dict = planning_batch_inference(
            data_samples=data_samples, 
            planner_model_id=self.model_name, 
            data_path=data_path, 
            save_path=save_path,
            use_local_planner=False,
            args=args,
        )
        return planning_traj_dict