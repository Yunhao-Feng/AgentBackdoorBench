# Backdoor Attacks and Defenses in LLM-Based Agents

This document reviews recent research (2022–2025) on **backdoor attacks** and **defenses** targeting **LLM-based agents**, focusing on **memory poisoning**, **tool usage**, **multi-step reasoning**, and **agent planning**. The aim is to provide an overview of backdoor threats within the context of agents that integrate LLMs as core components for reasoning, memory, and task execution.

## Table of Contents
- [Backdoor Attacks and Defenses in LLM-Based Agents](#backdoor-attacks-and-defenses-in-llm-based-agents)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Stage 1: Perception and Input Parsing](#stage-1-perception-and-input-parsing)
    - [Possible Backdoor Attacks:](#possible-backdoor-attacks)
    - [Representative Research:](#representative-research)
  - [Stage 2: Intent Recognition and Task Planning](#stage-2-intent-recognition-and-task-planning)
    - [Possible Backdoor Attacks:](#possible-backdoor-attacks-1)
    - [Representative Research:](#representative-research-1)
  - [Stage 3: Knowledge Retrieval and Reasoning](#stage-3-knowledge-retrieval-and-reasoning)
    - [Possible Backdoor Attacks:](#possible-backdoor-attacks-2)
    - [Representative Research:](#representative-research-2)
  - [Stage 4: Decision Generation and Execution](#stage-4-decision-generation-and-execution)
    - [Possible Backdoor Attacks:](#possible-backdoor-attacks-3)
    - [Representative Research:](#representative-research-3)
  - [Stage 5: Feedback and Learning Update](#stage-5-feedback-and-learning-update)
    - [Possible Backdoor Attacks:](#possible-backdoor-attacks-4)
    - [Representative Research:](#representative-research-4)
  - [Defensive Strategies](#defensive-strategies)
    - [1. **Execution Trace Validation and Self-Consistency Checks**](#1-execution-trace-validation-and-self-consistency-checks)
    - [2. **Robust Retrieval and Memory Sanitization**](#2-robust-retrieval-and-memory-sanitization)
  - [Conclusion](#conclusion)
  - [References](#references)

## Introduction

LLM-based agents integrate **large language models** with reasoning and task-execution modules, enabling complex behaviors beyond simple language generation. These systems are used in various applications, from autonomous systems to interactive AI. However, their complexity also introduces new backdoor vulnerabilities, which attackers can exploit to manipulate agent behavior at different stages.

This document organizes the backdoor attacks and defenses into **stages of agent behavior**, with a focus on attacks that target **memory poisoning**, **multi-step reasoning**, and **tool/knowledge interaction**.

---

## Stage 1: Perception and Input Parsing

**Stage Definition:** This stage involves the agent receiving and parsing input data from the user or environment. This includes natural language queries, observations, and context provided to the agent.

### Possible Backdoor Attacks:
- **Trigger Injection**: Insert malicious tokens or phrases into input data that activate a backdoor when parsed.
- **Covert Instructions**: Embed hidden instructions that influence the agent’s decision-making process when parsed.
- **Environmental Triggering**: Manipulate external data (e.g., web content, sensor inputs) to induce specific agent behavior.

### Representative Research:

| Paper Title | Conference (Year) | Type | Summary | Link |
| ----------- | ------------------ | ---- | ------- | ---- |
| **AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases** | NeurIPS 2024 | Attack | First backdoor attack on retrieval-augmented LLM agents by poisoning memory or knowledge bases. Malicious memory entries are injected to alter agent behavior when queried. | [Link](https://arxiv.org/abs/2402.06545) |
| **TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in LLMs** | ArXiv 2024 | Attack | Proposes corrupting a few knowledge entries in the retrieval corpus, using a "trigger-context" pair to induce backdoor behavior in RAG-based agents. | [Link](https://arxiv.org/abs/2405.13401) |


---

## Stage 2: Intent Recognition and Task Planning

**Stage Definition:** After parsing the input, the agent identifies the user’s intent and plans the appropriate actions to achieve the goal. This may involve breaking down tasks and interacting with external tools.

### Possible Backdoor Attacks:
- **Instruction Tuning Poisoning**: Attack the agent’s fine-tuning process with poisoned instruction data to mislead intent recognition.
- **Hidden Instruction Triggers**: Craft subtle triggers in the task instructions that alter the agent's understanding.
- **Planning Disruption**: Interfere with the agent’s planning algorithm to cause malicious behavior.

### Representative Research:

| Paper Title | Conference (Year) | Type | Summary | Link |
| ----------- | ------------------ | ---- | ------- | ---- |
| **BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents** | ACL 2024 | Attack | Demonstrates poisoning fine-tuning data to inject backdoors into the agent’s planning and reasoning module. Malicious behavior is activated by hidden triggers in user input or environmental conditions. | [Link](https://arxiv.org/abs/2306.09578) |
| **BadChain: Backdoor Chain-of-Thought Prompting for LLMs** | ICLR 2024 | Attack | Alters a few-shot chain-of-thought reasoning process by inserting malicious steps that activate on specific triggers. | [Link](https://openreview.net/forum?id=Zg9tN6f8GL) |

---

## Stage 3: Knowledge Retrieval and Reasoning

**Stage Definition:** The agent retrieves knowledge or information from external sources, performs reasoning, and derives actions or decisions based on that knowledge.

### Possible Backdoor Attacks:
- **Observation Triggering**: Manipulate the agent's knowledge retrieval or sensor data to induce backdoor behavior.
- **Poisoning Reasoning Process**: Alter intermediate steps in reasoning chains to cause the agent to deviate from intended outcomes.
- **Tool/Plugin Malicious Use**: Hijack tools or external plugins the agent uses to perform tasks.

### Representative Research:

| Paper Title | Conference (Year) | Type | Summary | Link |
| ----------- | ------------------ | ---- | ------- | ---- |
| **DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs** | ArXiv 2025 | Attack | Latent backdoors inserted into reasoning chains, activating without clear trigger in the prompt but based on intermediate reasoning steps. | [Link](https://arxiv.org/abs/2501.18617) |
| **Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents** | NeurIPS 2024 | Attack | Investigates multi-stage backdoor threats within LLM-based agents, including reasoning and tool-based manipulations. | [Link](https://arxiv.org/abs/2402.11208) |



---

## Stage 4: Decision Generation and Execution

**Stage Definition:** The agent generates the final decision or output based on the reasoning and performs the corresponding actions, such as answering queries, recommending actions, or executing operations.

### Possible Backdoor Attacks:
- **Output Manipulation**: Trigger backdoor to output harmful or malicious actions when specific conditions are met.
- **Harmful Content Injection**: The agent outputs dangerous or harmful responses under backdoor influence.
- **Erroneous Action Execution**: Backdoor that causes the agent to perform unintended actions, like unauthorized file deletion or malicious purchases.

### Representative Research:

| Paper Title | Conference (Year) | Type | Summary | Link |
| ----------- | ------------------ | ---- | ------- | ---- |
| **TrojText: Test-Time Invisible Textual Trojan Insertion** | ICLR 2023 | Attack | Inserts invisible syntactic triggers that alter the agent’s final output during testing, with no noticeable change to normal performance. | [Link](https://openreview.net/forum?id=Zg9tN6f8GL) |
| **A Watermark for Large Language Models** | ICML 2023 | Defense | Introduces a watermark mechanism to detect and prevent output manipulation by backdoor attacks. | [Link](https://arxiv.org/abs/2306.07740) |

---

## Stage 5: Feedback and Learning Update

**Stage Definition:** The agent learns from user feedback or experiences, updating its internal models or strategies to improve performance.

### Possible Backdoor Attacks:
- **Online Training Poisoning**: Introduce malicious feedback to influence the agent’s future decisions.
- **Model Update Manipulation**: Hijack the agent's learning process during model updates to inject malicious behavior.
- **Feedback Triggering**: Use specific feedback to activate the agent’s backdoor, leading to harmful actions.

### Representative Research:

| Paper Title | Conference (Year) | Type | Summary | Link |
| ----------- | ------------------ | ---- | ------- | ---- |
| **ReAgent: Your Agent Can Defend Itself against Backdoor Attacks** | ACL 2024 | Defense | Proposes a self-consistency defense where agents detect discrepancies between planned and executed tasks, preventing backdoor activation. | [Link](https://arxiv.org/abs/2405.11201) |
| **LT-Defense: Searching-free Backdoor Defense via Exploiting the Long-tailed Effect** | NeurIPS 2024 | Defense | Defense mechanism that utilizes long-tailed data distributions to detect and mitigate backdoor attacks. | [Link](https://arxiv.org/abs/2402.07830) |

---

## Defensive Strategies

Defending against backdoors in LLM-based agents is particularly challenging due to their complex multi-step reasoning and tool interactions. Here are some strategies that have been proposed:

### 1. **Execution Trace Validation and Self-Consistency Checks**
- *ReAgent* proposes using **execution-level verification** to catch discrepancies in reasoning and planning steps that result from a backdoor.
  
### 2. **Robust Retrieval and Memory Sanitization**
- *RobustRAG* introduces a memory sanitization technique that verifies and cross-checks external knowledge before it's used by the agent, ensuring that poisoned information doesn't influence decision-making.

---

## Conclusion

LLM-based agents are increasingly used in a wide variety of applications, but their complexity makes them vulnerable to novel types of **backdoor attacks**. These attacks can target different stages of the agent’s behavior, from perception and input parsing to decision execution and feedback learning. Research has shown that traditional backdoor detection and defense methods for LLMs are insufficient when applied to multi-step agents. Instead, **robust reasoning** and **memory validation** techniques are crucial for defending these agents against increasingly sophisticated attacks.

As the field evolves, we anticipate further development in **self-consistency checks**, **execution trace validation**, and **knowledge retrieval defenses** to secure LLM-based agents in real-world scenarios.

---

## References

- Z. Chen et al., “AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases,” NeurIPS 2024. [Link](https://arxiv.org/abs/2402.06545)
- P. Cheng et al., “TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in LLMs,” arXiv 2024. [Link](https://arxiv.org/abs/2405.13401)
- Z. Xiang et al., “BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents,” ACL 2024. [Link](https://arxiv.org/abs/2306.09578)
- W. Yang et al., “Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents,” NeurIPS 2024. [Link](https://arxiv.org/abs/2402.11208)
- E. Hubinger et al., “Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training,” Anthropic, ArXiv 2024. [Link](https://arxiv.org/abs/2401.05566)
- Y. Wang et al., “ReAgent: Your Agent Can Defend Itself against Backdoor Attacks,” ACL 2024. [Link](https://arxiv.org/abs/2405.11201)
- C. Xiang et al., “Certifiably Robust RAG against Retrieval Corruption,” arXiv 2024. [Link](https://arxiv.org/abs/2405.15556)
