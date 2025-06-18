import json
import os
import random
import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="MATH")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--num2sample", type=int, default=5)
args = parser.parse_args()

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{args.dataset_name}.json")

def shuffle_and_sample(data_list, num2sample):
    random.seed(2024)
    random.shuffle(data_list)
    return data_list[:num2sample]

def deduplicate(data_list):
    seen_queries = set()
    unique_data = []

    for item in data_list:
        if item["query"] not in seen_queries:
            unique_data.append(item)  # 添加第一个出现的样本
            seen_queries.add(item["query"])  # 标记这个 query 已经出现
    if len(unique_data) < len(data_list):
        print(f">> Duplicate samples removed: {len(data_list) - len(unique_data)}")
    return unique_data

# load MATH-500 dataset
if args.dataset_name == "MATH":
    load_dataset_path = args.dataset_path if args.dataset_path else "HuggingFaceH4/MATH-500"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": example["answer"],
            "tag": [args.dataset_name, "math", example["subject"], f"Level {example['level']}", ],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load GSM8K dataset
elif args.dataset_name == "GSM8K":
    load_dataset_path = args.dataset_path if args.dataset_path else "openai/gsm8k"
    dataset = load_dataset(load_dataset_path, "main", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["question"],
            "gt": example["answer"],
            "tag": ["math"],
            "source": "GSM8K"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load AQUA-RAT dataset
elif args.dataset_name == "AQUA-RAT":
    load_dataset_path = args.dataset_path if args.dataset_path else "deepmind/aqua_rat"
    dataset = load_dataset(load_dataset_path, "raw", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)   # question / options / rationale / correct
    def format_aqua_rat_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for option in example["options"]:
            query += f"\n{option}"
        return query
    data_list = [
        {
            "query": format_aqua_rat_query(example),
            "gt": example["rationale"],
            "tag": ["math", "reasoning", "multiple-choice"],
            "source": "AQUA-RAT"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MedMCQA
elif args.dataset_name == "MedMCQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "openlifescienceai/medmcqa"
    dataset = load_dataset(load_dataset_path, split="validation", trust_remote_code=True)
    filtered_dataset = dataset.filter(lambda example: example['choice_type'] != 'multi')
    print(f"{'='*50}\n", filtered_dataset)
    def format_medmcqa_query(example):
        query = example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['opa']}"
        query += f"\n(B) {example['opb']}"
        query += f"\n(C) {example['opc']}"
        query += f"\n(D) {example['opd']}"
        return query
    def format_medmcqa_gt(example):
        answer_list = [f"(A) {example['opa']}", f"(B) {example['opb']}", f"(C) {example['opc']}", f"(D) {example['opd']}"]
        answer = f"The correct answer is: {answer_list[example['cop']]}"
        return answer
    data_list = [
        {
            "query": format_medmcqa_query(example),
            "gt": format_medmcqa_gt(example),
            "tag": ["medical", example['subject_name'], example['topic_name']],
            "source": "MedMCQA"
        }
        for example in filtered_dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MedQA
elif args.dataset_name == "MedQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "bigbio/med_qa"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_medqa_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for option in example["options"]:
            query += f"\n({option['key']}) {option['value']}"
        return query
    def format_medqa_gt(example):
        answer = f"The correct answer is: ({example['answer_idx']}) {example['answer']}"
        return answer
    data_list = [
        {
            "query": format_medqa_query(example),
            "gt": format_medqa_gt(example),
            "tag": ["medical"],
            "source": "MedQA"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MMLU
elif args.dataset_name == "MMLU":
    load_dataset_path = args.dataset_path if args.dataset_path else "cais/mmlu"
    dataset = load_dataset(load_dataset_path, "all", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def format_mmlu_query(example):
        query = f"""The following is a multiple-choice question:
{example["question"]}

Choose the correct answer from the following options:
(A) {example["choices"][0]}
(B) {example["choices"][1]}
(C) {example["choices"][2]}
(D) {example["choices"][3]}"""
        return query
    
    def format_mmlu_gt(example):
        choice_list = ["A", "B", "C", "D"]
        answer = f"({choice_list[example['answer']]})"
        return answer
    
    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["mmlu", example['subject']],
            "source": "MMLU"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MMLU-Pro
elif args.dataset_name == "MMLU-Pro":
    load_dataset_path = args.dataset_path if args.dataset_path else "TIGER-Lab/MMLU-Pro"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def format_mmlu_query(example):
        query = "The following is a multiple-choice question:\n"
        query += example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        for idx, option in enumerate(example["options"]):
            query += f"\n({option_list[idx]}) {option}"
        return query
    
    def format_mmlu_gt(example):
        answer = f"The answer is ({option_list[example['answer_index']]}) {example['options'][example['answer_index']]}"
        return answer
    
    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["MMLU-Pro", example['category'], example['src']],
            "source": "MMLU-Pro",
            "num_choices": len(example["options"])
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)  # 1001, 2004

# load GSM-Hard dataset
elif args.dataset_name == "GSM-Hard":
    load_dataset_path = args.dataset_path if args.dataset_path else "reasoning-machines/gsm-hard"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["input"],
            "gt": str(example["target"]),
            "tag": ["math", "GSM-Hard"],
            "source": "GSM-Hard"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load GPQA dataset
elif args.dataset_name.startswith("GPQA"):
    load_dataset_path = args.dataset_path if args.dataset_path else "Idavidrein/gpqa"
    if args.dataset_name == "GPQA-Diamond":
        dataset = load_dataset(load_dataset_path, "gpqa_diamond", split="train", trust_remote_code=True)
    else:
        dataset = load_dataset(load_dataset_path, "gpqa_main", split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_gpqa_query(example):
        query = example["Question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['Correct Answer']}"
        query += f"\n(B) {example['Incorrect Answer 1']}"
        query += f"\n(C) {example['Incorrect Answer 2']}"
        query += f"\n(D) {example['Incorrect Answer 3']}"
        return query
    def format_gpqa_gt(example):
        answer = f"(A) {example['Correct Answer']}"
        return answer
    data_list = [
        {
            "query": format_gpqa_query(example),
            "gt": format_gpqa_gt(example),
            "tag": [args.dataset_name, example["High-level domain"], example["Subdomain"], example["Writer's Difficulty Estimate"]],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load SciBench dataset
elif args.dataset_name == "SciBench":
    load_dataset_path = args.dataset_path if args.dataset_path else "xw27/scibench"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_scibench_gt(example):
        answer = f"{example['answer_number']}, the unit is {example['unit']}."
        return answer
    data_list = [
        {
            "query": example['problem_text'],
            "gt": format_scibench_gt(example),
            "tag": [args.dataset_name, 'science', example['source']],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

elif args.dataset_name == "AIME-2024":
    load_dataset_path = args.dataset_path if args.dataset_path else "Maxwell-Jia/AIME_2024"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["Problem"],
            "gt": example["Answer"],
            "tag": [args.dataset_name, "math"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

else:
    raise ValueError(f"Dataset {args.dataset_name} not supported.")

sample_pool = deduplicate(data_list)
print(f">> A data sample from the pool:\n{sample_pool[0]}")

print(f"{'='*50}\n There are {len(sample_pool)} queries in the pool.")

with open(save_path, 'w') as output_json:
    json.dump(sample_pool, output_json, indent=4)