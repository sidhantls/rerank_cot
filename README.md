# rerank_cot

A document re-ranking evaluation framework that explores whether reasoning tokens can improve re-ranking performance. The framework compares a baseline approach against a reasoning version that prompts the Qwen Re-ranker model to first predict an analysis of query-document similarity before making relevance predictions. 

The motivation is that explicit reasoning steps has helped smaller LLMs, similar to CoT's success.

## Overview

This project evaluates document re-ranking performance using two approaches:
- **Baseline**: Standard re-ranking without explicit reasoning
- **Reasoning**: Re-ranking with chain-of-thought reasoning where Qwen outputs its reasoning before making the final score prediction

## Running Evaluation

To evaluate the re-ranker performance:

```bash
python evaluate.py
```

Update arguments in args as per the requirements.

### Configuration

The evaluation script (`evaluate.py`) supports the following configurations:

- **Dataset**: Currently supports "jinaai/hotpotqa-reranking-en"
- **Re-ranker Type**: 
  - `"baseline"`: Standard re-ranking implementation
  - `"reasoning"`: Reasoning-based re-ranking with CoT
- **Model**: Qwen model for re-ranking (default: "Qwen/Qwen3-Reranker-0.6B")
- **Batch Size**: Processing batch size (default: 8)
- **Max Samples**: Maximum samples to evaluate (default: 500)

### Experimental Results

This section compares two re-ranking approaches: the baseline approach that directly predicts relevance scores, and the reasoning approach that generates chain-of-thought explanations before making predictions. We evaluate both methods on document re-ranking tasks to understand the impact of explicit reasoning on performance.

**Experiment 1: Reasoning-based Re-ranking**

**Results**:
| Metric | With Reasoning | Baseline |
|--------|---------------|----------|
| Top-1 Accuracy | 0.4900 | 0.9710 |
| Top-3 Accuracy | 0.6300 | 0.9960 |
| Top-5 Accuracy | 0.7100 | 1.0000 |

**Why the reasoning approach didn't work**: The Qwen reranker was fine-tuned with only predicting "yes" or "no" responses using a specific system prompt. However, when we changed the input format and system prompt to include reasoning steps, the model's performance significantly degraded because it was not trained to handle this new format. The model's fine-tuning was optimized for the original task structure, making it less effective when the system prompt has been changed. 

**Prompts Used**:

The reasoning approach used the following system prompt (see detailed prompt below):

```
prefix = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. First, provide a brief reasoning about the similarity between the query and document, and whether the query can be answered by the document. Then provide your final answer as "yes" or "no".

Examples:

Query: "What is the population of Tokyo?"
Document: "Tokyo is the capital city of Japan with a population of approximately 14 million people in the metropolitan area."
Reasoning: The query asks specifically for Tokyo's population, and the document directly provides this information stating "approximately 14 million people." The document perfectly matches the query's information need.
Final Answer: yes

..

"""
```

