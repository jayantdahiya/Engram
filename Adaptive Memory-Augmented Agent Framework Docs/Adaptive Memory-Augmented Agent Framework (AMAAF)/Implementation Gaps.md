# Implementation Gaps

---

Implementation gaps that the original v1.0 of this system doesn’t include. To be included in the future v2.0 version.

## Mem0-specific gaps

- Asynchronous summary refresher: No explicit background module that periodically generates and refreshes the conversation summary S used in extraction; Mem0 specifies an independent async summary generation component to keep S current without blocking the pipeline. [1][2]
- Explicit recency window orchestration: No concrete parameterization and scheduling for the m-length recent message sliding window in prompts P=(S,{m_{t-m}..m_{t-2}},m_{t-1},m_t); the plan implies recency but does not codify this rolling slice as a maintained input to extraction. [1]
- Tool-call style operation selection prompt: The plan has the ADD/UPDATE/DELETE/NOOP logic, but there is no explicit function-calling “tool” schema and prompt template for the LLM to decide operations against top-s similar memories as described; this includes feeding the s retrieved candidates to the LLM for operation selection. [1]
- Asynchronous update mechanics: The plan includes Celery, but does not specify that memory updates are performed asynchronously relative to request inference to mirror Mem0’s “non-blocking” extraction/update path and p95 latency goals. [2][1]
- Retrieval parameterization: No explicit configuration for “s = 10 similar memories” in the update comparison step and consistent use of this parameter across services. [1]

## Mem0g (graph memory) gaps

- Conflict detection and invalidation: The plan lacks an explicit conflict detector and an “invalidate but not hard-delete” mechanism for relationships to support temporal reasoning as in Mem0g. [1]
- Dual retrieval for graphs: No entity-centric subgraph expansion and no semantic triplet matching path; only basic create/merge operations are outlined without the two retrieval modes. [1]
- Node similarity thresholding: No defined similarity threshold t for node reuse vs creation during triplet ingestion. [1]
- Metadata policy: No schema for labels L, entity types, timestamps tv, and relation metadata required by Mem0g’s directed labeled graph design. [1]

## Memory‑R1 (RL) gaps

- PPO/GRPO training loops: No reinforcement learning pipelines (PPO/GRPO) for Memory Manager or Answer Agent, including actor/critic or reference policy handling, clipping, KL penalty, or grouped sampling. [3][4][5]
- Reward wiring: No outcome-driven reward computation tied to downstream question answering (e.g., Exact Match with gold, or J-based) and no data flow from Answer Agent outcomes back to Memory Manager policy updates. [3][4]
- Data construction for RL: No procedures to build tuples combining temporal memory banks, turns, and QA pairs for policy training, nor the retrieval-of-60-candidate-memories protocol used for Answer Agent training. [3]
- Memory distillation as policy: While a heuristic distillation step is present, there is no learned selection policy trained with PPO/GRPO for filtering retrieved memories before answering. [3]
- Training infrastructure: No mention of a framework (e.g., VERL) or GPU training orchestration for RL fine-tuning; no checkpoints, curriculum, or evaluation loop for RL runs. [3]

## ACAN (LLM-trained cross attention) gaps

- LLM-assisted training loop: The plan omits the LLM-scored training procedure that compares ACAN-selected memories to a baseline set and optimizes a custom loss using LLM-generated scores. [6]
- Custom loss function: No implementation of the loss that normalizes the LLM score difference and trains the attention network accordingly. [6]
- Learned attention model: Retrieval uses engineered composite scoring; there is no end-to-end training of a query-key-value cross-attention model whose attention weights rank memories, as proposed. [6]
- Evaluation harness: No LLM-based evaluation pipeline for retrieval quality during ACAN training or ablations against baseline retrieval. [6]

## Evaluation and metrics gaps (cross-paper)

- LLM-as-a-Judge integration: No automated J evaluation pipeline over question sets and no multi-run averaging with variance reporting as used in Mem0 and Memory‑R1. [1][3]
- Token budget tracking: No token accounting for memory-store footprint and retrieval context tokens to reproduce efficiency metrics. [1]
- Latency benchmarking: No p50/p95 benchmarking harness separating search latency vs total latency to validate targets and compare variants. [1]

## Prompting and datasets gaps

- Canonical prompts: No inclusion of the specific Mem0/Memory‑R1 prompt templates for operation selection, memory-augmented answering, or judge prompts in the codebase. [1][3]
- Candidate count protocol: No retrieval-of-60-candidates convention for the Answer Agent prior to distillation, as specified by Memory‑R1. [3]

These gaps are the remaining items needed to align the implementation with the three papers’ core contributions and evaluation methodologies. [1][3][6]

Sources
[1] Mem0: Building Production-Ready AI Agents with Scalable ... [https://arxiv.org/html/2504.19413v1](https://arxiv.org/html/2504.19413v1)
[2] How Mem0 Lets LLMs Remember Everything Without ... [https://apidog.com/blog/mem0-memory-llm-agents/](https://apidog.com/blog/mem0-memory-llm-agents/)
[3] Memory-R1: Enhancing Large Language Model Agents to ... [https://arxiv.org/pdf/2508.19828.pdf](https://arxiv.org/pdf/2508.19828.pdf)
[4] [Literature Review] Memory-R1: Enhancing Large ... [https://www.themoonlight.io/review/memory-r1-enhancing-large-language-model-agents-to-manage-and-utilize-memories-via-reinforcement-learning](https://www.themoonlight.io/review/memory-r1-enhancing-large-language-model-agents-to-manage-and-utilize-memories-via-reinforcement-learning)
[5] Memory-R1: Reinforced Memory for LLMs [https://www.emergentmind.com/topics/memory-r1](https://www.emergentmind.com/topics/memory-r1)
[6] Enhancing memory retrieval in generative agents through LLM ... [https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full)
[7] 2504.19413v1.pdf [https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/9403925/cb250ade-99e4-4554-82be-3ed9b827983e/2504.19413v1.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/9403925/cb250ade-99e4-4554-82be-3ed9b827983e/2504.19413v1.pdf)
[8] Introduction - Mem0 [https://docs.mem0.ai](https://docs.mem0.ai/)
[9] Overview [https://docs.mem0.ai/open-source/features/overview](https://docs.mem0.ai/open-source/features/overview)
[10] Mem0: Building Production-Ready AI Agents with Scalable ... [https://arxiv.org/abs/2504.19413](https://arxiv.org/abs/2504.19413)
[11] Advanced Memory Operations [https://docs.mem0.ai/platform/advanced-memory-operations](https://docs.mem0.ai/platform/advanced-memory-operations)
[12] Enhancing Code LLM Training with Programmer Attention - arXiv [https://arxiv.org/html/2503.14936v2](https://arxiv.org/html/2503.14936v2)