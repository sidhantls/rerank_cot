# rerank_cot

A document re-ranking evaluation framework that compares baseline and reasoning-based approaches using the Qwen Re-ranker model.

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
