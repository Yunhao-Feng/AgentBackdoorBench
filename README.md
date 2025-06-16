# 🔐 Agent Backdoor Attack Benchmark

This repository provides a **categorized benchmark** for **backdoor attacks targeting LLM-based agents**. With the rapid adoption of LLM agents in applications like finance, healthcare, and web automation, understanding their unique security vulnerabilities is crucial for safe deployment.

---

## 📊 Attack Taxonomy

We classify current backdoor attacks into **four major categories**, based on the *attack surface*, *trigger mechanism*, and *agent behavior manipulation*:

| 🧩 Category | 🔍 Attack Surface | 🎯 Trigger Type | 📚 Representative Works | 🧠 Key Characteristics | 🛡 Defense Challenges |
|------------|------------------|----------------|-------------------------|------------------------|-----------------------|
| **1. Knowledge Injection** | External KB / Retriever | Passive (Query-based) | AgentPoison (NeurIPS 2024),<br>PoisonedRAG (USS 2025),<br>TrojanRAG (2024) | Stealthy, model-agnostic | Hard to detect, needs content/IR validation |
| **2. Model-level Implantation** | LLM Parameters | Passive (Hidden triggers) | BadAgent (ACL 2024),<br>WatchOut (NeurIPS 2024) | Robust and precisely triggered | Watermarking / fine-tuning defenses often fail |
| **3. Reasoning Manipulation** | Prompt / CoT Context | In-context trigger | BadChain (ICLR 2024),<br>DemonAgent (2025) | Lightweight, stealthy | Requires trace-level consistency checking |
| **4. Active Adversarial Agents** | Full agent behavior / tools | Active (Query / Env / Obs) | PI AdvAgent (ICML 2025) | Multi-modal tool-level attacks | Costly to simulate and hard to generalize defenses |

---

## 🧪 Benchmark Goals

This benchmark aims to:

✅ Systematically categorize agent backdoor threats  
✅ Collect representative attacks and implementations  
✅ Define standard evaluation metrics (ASR, stealth, transferability)  
✅ Facilitate development of effective, agent-specific defenses

---

## 📂 Included Works & Codebases

| Paper | Code / Link |
|-------|-------------|
| AgentPoison (NeurIPS 2024) | [GitHub](https://github.com/BillChan226/AgentPoison) |
| PoisonedRAG | [GitHub](https://github.com/sleeepeer/PoisonedRAG) |
| TrojanRAG | [GitHub](https://github.com/Charles-ydd/TrojanRAG) |
| BadAgent (ACL 2024) | [GitHub](https://github.com/DPamK/BadAgent) |
| BadChain (ICLR 2024) | [arXiv](https://arxiv.org/abs/2401.12242) |
| AdvAgent (ICML 2025) | [Website](https://ai-secure.github.io/AdvAgent) |
| DemonAgent | [GitHub](https://github.com/whfeLingYu/DemonAgent) |

---

## 🛡️ Defense Landscape

| Defense | Description | Highlights |
|---------|-------------|------------|
| **ShieldAgent** (2025) | Uses verifiable reasoning-based safety policies to monitor and block harmful agent actions. | - Model-agnostic logic-based verification <br> - Captures policy violations via trace audit |
| Prompt Filtering | Heuristic-based rejection of suspicious inputs or tools. | Weak against stealthy / CoT-based attacks |
| Safety-Aware Memory | Blacklisting memory traces post detection. | Not effective against dynamically encrypted payloads (e.g., DemonAgent) |

⚠️ **Observation**: Most existing defenses are *insufficient* against dynamic, multi-step, or reasoning-injected backdoors.

---

## 🧠 Common Agent Architectures

### 🌐 WebAgent-based Agents

- Use LLMs as controllers to browse real websites, perform form filling, click buttons, and submit inputs.
- Common components:
  - Environment parser (DOM extractor)
  - Task planner (LLM)
  - Action executor (browser API)
- **Targeted by**: AdvAgent (black-box HTML injection), DemonAgent (multi-hop manipulation)

### 🔁 ReAct-style Agents

- Use iterative *Thought → Action → Observation* loops, inspired by Chain-of-Thought prompting.
- Components:
  - Thought: reasoning steps
  - Action: tool/API call
  - Observation: result from environment
- **Targeted by**: BadChain (COT poisoning), WatchOut (observation manipulation)

🧩 These two architectures define **key attack surfaces** for backdoor triggering and planning logic manipulation.

---

## 🧠 Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **ASR** (Attack Success Rate) | Percentage of queries where the backdoor is triggered and achieves malicious goal |
| **Stealth** | Attack visibility to human or detector |
| **Robustness** | Success rate under input variations or defense presence |
| **Transferability** | Effectiveness across models / agent tasks |
| **Trigger Compactness** | How small and hidden the trigger can be |

---

## 📌 Citation

Please cite individual works as needed. A collective survey paper and BibTeX export will be released soon.

---

## 🤝 Contribution Guidelines

We welcome contributions on:

- 📌 New backdoor attacks or variants  
- 🧪 Agent-specific defenses  
- 📁 Dataset proposals for realistic attack simulation  
- 🔧 Benchmark pipeline / evaluation scripts

Please open an issue or PR to get involved.
