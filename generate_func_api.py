import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import argparse
import random
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from utils.load_model_commands import load_model
from utils.attack_commands import load_attack
from utils.detect_commands import VoteTRANS
from utils.load_dataset_commands import load_dataset_from_huggingface
from utils.data_commands import read_data, save_data
from textattack.attack_results import SkippedAttackResult
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score


from textattack import Attacker
from textattack import AttackArgs
ADV_LABEL = 1
ORG_LABEL = 0
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


from flask import Flask,request,jsonify,send_file

app = Flask(__name__)

@app.route('/')
def test():
    print("--API-对抗样本检测功能 ")
    return 'aaaaaa'


def evaluate(org_texts, adv_texts, target, supports, auxidiary_attack, word_ratio = 1.0):    
    """ evaluate VoteTRANS for balanced texts of original and adversarial texts
    Args:        
        org_texts (list of string):
            list of original texts
        adv_texts (list of string):
            list of adversarial texts
        target:
            an target model loaded by TextAttack
        supports:
            support models loaded by TextAttack
        auxidiary_attack:
            an attack from TextAttack.
        word_ratio (float, default = 1.0):
            a threshold to determine the number of words in the input text are used to process        
    Returns:      
        f1 (float):
            F1-score
        recall (float):
            Adverarial recall
    """  
    cln_text, cln_pred = [], []
    adv_text, adv_pred = [], []
    gold_labels = []
    detect_labels = []
    num_pairs = len(org_texts)
    for index in range(num_pairs):    
#        detect original text
        detect_result = VoteTRANS(org_texts[index], target, supports, auxidiary_attack, word_ratio = word_ratio)            
        gold_labels.append(ORG_LABEL)
        detect_labels.append(detect_result)
        
#        detect adversarial text
        detect_result = VoteTRANS(adv_texts[index], target, supports, auxidiary_attack, word_ratio = word_ratio)            
        gold_labels.append(ADV_LABEL)
        detect_labels.append(detect_result)       
        
#        print results
        org_result = "CORRECT" if (gold_labels[len(gold_labels) - 2] == detect_labels[len(detect_labels) - 2]) else "INCORRECT" 
        adv_result = "CORRECT" if (gold_labels[len(gold_labels) - 1] == detect_labels[len(detect_labels) - 1]) else "INCORRECT"
        
        print(f"Pair {index + 1} / {len(org_texts)} : Original detection : {org_result} ; Adversarial detection = {adv_result}")
        org_result = "Clean" if (gold_labels[len(gold_labels) - 2] == detect_labels[len(detect_labels) - 2]) else "Adversarial" 
        adv_result = "Adversarial" if (gold_labels[len(gold_labels) - 1] == detect_labels[len(detect_labels) - 1]) else "Clean"
        cln_text.append(org_texts[index])
        adv_text.append(adv_texts[index])

        cln_pred.append(org_result)
        adv_pred.append(adv_result)

        
    f1 = f1_score(gold_labels, detect_labels, average = 'binary')    
    recall = recall_score(gold_labels, detect_labels, average = 'binary')
    accuracy = accuracy_score(gold_labels, detect_labels)
    auroc = roc_auc_score(gold_labels, detect_labels)
    
    fpr, tpr, thresholds = roc_curve(gold_labels, detect_labels)
    roc_auc = roc_auc_score(gold_labels, detect_labels)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # 保存图像到本地文件
    plt.savefig('ROC_curve.png')
    res = pd.DataFrame({
        'cln_text':cln_text,
        "cln_pred":cln_pred,
        'adv_text':adv_text,
        'adv_pred':adv_pred
    })
    
    return f1, recall, accuracy, auroc, res

#输入参数   数据集路径   保存输出图片路径  攻击方法
@app.route('/txt_attack_generate_api', methods=['POST'])
def txt_attack_generate_api(model_name="cnn-ag-news", dataset_name="ag_news", attack_src="pwws", num_pairs=10, output_path='data.pkl'):
    """
    生成函数
    num_pairs:要生成的对抗样本数量。
    output:将结果保存为pkl。
    attack:攻击方法。pwws/pso
    """
    target = load_model(model_name=model_name)
    supporter_name = 'roberta-base-ag-news'
    supports = []
    supports.append(load_model(supporter_name))
    attack = load_attack(attack_src, target)

    auxidiary_attack = load_attack(attack_src, target)
    dataset = load_dataset_from_huggingface(dataset_name)
    attack_args = AttackArgs(num_successful_examples=num_pairs)
    attacker = Attacker(attack, dataset, attack_args)
    attack_results = attacker.attack_dataset() # attack
    save_data(dataset, attack_results, output_path)
    print(attack_results)
    return [str(i) for i in attack_results]

@app.route('/txt_attack_text_api', methods=['POST'])
def txt_attack_text_api(model_name="cnn-ag-news",pkl_path="data.pkl", attack="pwws"):
    """检测函数"""
    org_texts, adv_texts, _ = read_data(pkl_path)
    target = load_model(model_name=model_name)
    supporter_name = 'roberta-base-ag-news'
    supports = []
    supports.append(load_model(supporter_name))
    attack = load_attack(attack, target)

    auxidiary_attack = load_attack("pwws", target)

    f1, recall,acc,auroc, res = evaluate(org_texts, adv_texts, target, supports, auxidiary_attack)
    # return [f1, recall,acc,auroc, res]
    return [f1, recall,acc,auroc, res.to_string()]


# if __name__ == "__main__":
    # res = txt_attack_generate_api()
    # print(res)
    # test_res = txt_attack_text_api()
    # print(test_res)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=48096,debug=True)