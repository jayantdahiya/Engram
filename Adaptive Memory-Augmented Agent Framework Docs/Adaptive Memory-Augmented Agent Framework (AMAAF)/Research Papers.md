# Research Papers

---

### 1. **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**

**Paper:**

[2504.19413v1.pdf](Research%20Papers%202643d997fd1b8043a9c3e1a107c92012/2504.19413v1.pdf)

| **Aspect** | **Details** |
| --- | --- |
| **Authors** | Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav |
| **Publication** | arXiv:2504.19413v1, April 2025 |
| **Core Problem** | LLMs' fixed context windows limit coherent multi-session conversations |
| **Solution** | Two-phase memory architecture with extraction and update phases |

### **Key Components**

- **Mem0 Base**: Natural language memory with {ADD, UPDATE, DELETE, NOOP} operations
- **Mem0g**: Graph-based enhancement with entities as nodes and relationships as edges
- **Architecture**: Two-phase pipeline (extraction → update)
- **Database**: Vector embeddings with semantic similarity search

### **Performance Results**

- **Accuracy**: 26% improvement over OpenAI memory system
- **Efficiency**: 91% lower p95 latency, 90%+ token cost reduction
- **Memory Usage**: 7K tokens (base), 14K tokens (graph variant)

### **Strengths**

- Production-ready scalable design
- Excellent computational efficiency
- Clean operational paradigm
- Strong baseline performance

### **Limitations**

- Static memory operations (no learning)
- Limited adaptation to user patterns
- Basic retrieval mechanisms

---

### 2. **Memory-R1: Enhancing LLM Agents via Reinforcement Learning**

Paper:

[2508.19828v3.pdf](Research%20Papers%202643d997fd1b8043a9c3e1a107c92012/2508.19828v3.pdf)

| **Aspect** | **Details** |
| --- | --- |
| **Authors** | Sikuan Yan, Xiufeng Yang, Zuchao Huang, et al. |
| **Publication** | arXiv:2508.19828v3, September 2025 |
| **Core Innovation** | First RL framework for memory-augmented LLMs |
| **Architecture** | Dual-agent system with specialized Memory Manager and Answer Agent |

### **Key Components**

- **Memory Manager**: RL-trained for optimal {ADD, UPDATE, DELETE, NOOP} operations
- **Answer Agent**: Memory Distillation policy for filtering retrieved memories
- **Training**: PPO and GRPO with outcome-driven rewards
- **Data Efficiency**: Effective with as few as 152 training examples

### **Performance Results**

- **vs Mem0**: 48% F1 improvement, 69% BLEU-1 improvement, 37% LLM-as-a-Judge improvement
- **Cross-Model**: Works on both LLaMA-3.1-8B and Qwen-2.5-7B
- **Generalization**: Strong performance across all question types

### **Strengths**

- Adaptive learning through RL
- Memory Distillation reduces noise
- Data-efficient training
- Robust cross-model performance

### **Limitations**

- Requires training infrastructure
- Higher computational overhead during training
- Dependency on reward signal quality

---

### 3. **ACAN: Enhancing Memory Retrieval via Cross-Attention Networks**

Paper:

[fpsyg-16-1591618.pdf](Research%20Papers%202643d997fd1b8043a9c3e1a107c92012/fpsyg-16-1591618.pdf)

| **Aspect** | **Details** |
| --- | --- |
| **Authors** | Chuanyang Hong, Qingyun He |
| **Publication** | Frontiers in Psychology, May 2025 |
| **Core Innovation** | Cross-attention mechanism for dynamic memory retrieval |
| **Training Method** | LLM-assisted training with custom loss function |

### **Key Components**

- **ACAN Architecture**: Cross-attention network with query-key-value mechanism
- **Dynamic Ranking**: Attention weights determine memory relevance
- **LLM Training**: Uses LLMs to evaluate and shape loss function
- **Simulation Environment**: Text-based multi-agent community

### **Performance Results**

- **Memory Scores**: 5.94 average vs 5.05 baseline (WMR method)
- **Statistical Significance**: T-statistic 7.44, P-value 2.42×10⁻¹³
- **Behavioral Consistency**: 32.6% vs 24.6% event attendance rate
- **Robustness**: Lower standard deviation across trials

### **Strengths**

- Dynamic adaptation to context
- Human-like memory simulation
- Real-time responsiveness
- Novel LLM-assisted training

### **Limitations**

- High computational cost for LLM evaluation
- Limited scalability due to training complexity
- Dependency on simulation environment

---