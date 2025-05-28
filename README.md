# Awesome_AIAgent_Security
Paper of AI Agent Security


### SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference  
SparseVLM proposes a training-free, text-guided visual token pruning method that leverages cross-modal attention to adaptively sparsify vision-language models by selecting relevant text tokens to assess visual token importance and recycling pruned tokens to retain essential information, thereby significantly improving inference efficiency without sacrificing performance.  

### An Image is Worth 1/2 Tokens After Layer 2: Plug-and-PLay Acceleration for VLLM Inference (ECCV 2024)  
FastV addresses the inefficiency of visual token attention in large vision-language models by dynamically pruning redundant image tokens after early layers based on attention scores, offering a plug-and-play, training-free acceleration method that significantly reduces inference cost while preserving task performance.  

### AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases (NIPS 2024)
This paper introduces AGENTPOISON, a novel red-teaming method that executes backdoor attacks on retrieval-augmented generation (RAG)-based LLM agents by injecting a small number of optimized adversarial instances into their memory or knowledge base, using a constrained discrete optimization algorithm to craft stealthy triggers that force the retrieval and use of malicious demonstrations during inference—causing the agent to take harmful actions when the trigger is present, while preserving normal behavior otherwise.  

### SHIELDAGENT: Shielding Agents via Verifiable Safety Policy Reasoning  
The paper introduces SHIELDAGENT, an LLM-based guardrail agent that defends autonomous agents by explicitly verifying their action trajectories against structured safety policies using a novel action-based probabilistic safety policy model (ASPM), which is automatically constructed from policy documents via logical rule extraction, iterative refinement, and circuit-based clustering; during inference, SHIELDAGENT dynamically verifies only relevant action circuits using formal tools and a hybrid memory of workflows, enabling precise, efficient, and explainable shielding of agent behaviors.