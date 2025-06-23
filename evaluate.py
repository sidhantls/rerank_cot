import argparse
import json
import os
import time
from datasets import load_dataset
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Import the re-ranker models
from baseline import load_model_and_tokenizer as load_baseline_model, predict as baseline_predict
from reasoning import predict_with_reasoning
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
Prompt: Now, the goal is to evaluate the performance of a re-ranker (either baseline or reasoning). Refer to baseline.py and reasoning.py to see how the re-ranker can be used to re-rank the documents.

Now, you have loaded the docuemnts. Follow the instructions below to evaluate the re-ranker:
1. Add arguments of dataset name. Currently, only support "jinaai/hotpotqa-reranking-en".
2. Add arguments if we want to evaluate baseline or reasoning re-ranker.

After arguments are setup, use the re-ranker to re-rank the documents in the dataset. Utilize predict_with_reasoning from appropriate re-ranker file (baseline.py or reasoning.py) to get the scores for each document.
After this, calculate the top-k accuracy. Do k=1, 3, 5.

While doing this, i want to to also mainaint a log of (query, document, prob_score, reasoning). This is just logs for later analysis, which you can save to logs/.
After you calcualte the metrics, save it to logs/metrics.json.
In both these files, add a dataset_name key so i know where it comes from.

Standardize the filenames you use: model_name (after the "/"). and the either baseline or reasoning, depending on the re-ranker you are using.
"""


def parse_args():
    class Args:
        def __init__(self):
            self.dataset = "jinaai/hotpotqa-reranking-en"
            self.reranker = "reasoning"
            self.model = "Qwen/Qwen3-Reranker-0.6B"
            self.batch_size = 8 
            self.max_samples = 500 # max samples to evaluate on
            self.max_length = 512 # max length of the input tokens

    args = Args()
    return  args

def calculate_top_k_accuracy(scores: List[float], positive_count: int, ks: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """Calculate top-k accuracy for different k values"""
    # Create list of (score, is_positive) pairs
    score_pairs = [(scores[i], i < positive_count) for i in range(len(scores))]
    # Sort by score in descending order
    score_pairs.sort(key=lambda x: x[0], reverse=True)

    results = {}
    for k in ks:
        # Check if any positive document is in top-k
        top_k = score_pairs[:k]
        found_positive = any(is_positive for _, is_positive in top_k)
        results[f"top_{k}_accuracy"] = float(found_positive)
    return results

def create_log_entry(query: str, document: str, score: float, reasoning: Optional[str] = None) -> Dict[str, Any]:
    """Create a log entry for a document"""
    entry = {
        "query": query,
        "document": document,
        "prob_score": score,
    }
    if reasoning is not None:
        entry["reasoning"] = reasoning
    return entry

def evaluate_reranker(args):
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    dataset = dataset['test']

    # Create directory for logs if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Derive model short name for filenames
    model_short_name = args.model.split("/")[-1]
    reranker_type = args.reranker
    logs_filename = f"logs/{model_short_name}_{reranker_type}_logs.jsonl"
    metrics_filename = f"logs/{model_short_name}_{reranker_type}_metrics.json"

    # Load model based on reranker type
    print(f"Loading {reranker_type} re-ranker model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Limit samples if specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Prepare metrics tracking
    total_samples = len(dataset)
    correct = {1: 0.0, 3: 0.0, 5: 0.0}
    all_logs = []

    # Process each sample
    start_time = time.time()
    print(f"Evaluating on {total_samples} samples...")

    for idx, sample in enumerate(tqdm(dataset)):
        query = sample["query"]
        positive_docs = sample["positive"]
        negative_docs = sample["negative"]

        # Combine all documents (positive first, then negative)
        all_docs = positive_docs + negative_docs
        positive_count = len(positive_docs)

        # Get scores using the appropriate method
        if reranker_type == "baseline":
            scores = baseline_predict(query, all_docs, model, tokenizer, batch_size=args.batch_size,
                                      max_length=args.max_length)
            reasonings = [None] * len(all_docs)
        else:  # reasoning
            scores, reasonings = predict_with_reasoning(query, all_docs, model, tokenizer,
                                                        batch_size=args.batch_size,
                                                        max_length=args.max_length)

        # Create log entries for this sample
        for doc_idx, (doc, score, reasoning) in enumerate(zip(all_docs, scores, reasonings)):
            log_entry = create_log_entry(query, doc, score, reasoning)
            log_entry["is_positive"] = doc_idx < positive_count
            all_logs.append(log_entry)

        # Calculate accuracy for this sample
        sample_results = calculate_top_k_accuracy(scores, positive_count, ks=[1, 3, 5])

        # Update metrics
        for k in [1, 3, 5]:
            correct[k] += sample_results[f"top_{k}_accuracy"]

    total_time = time.time() - start_time

    # Calculate overall metrics
    metrics = {
        "dataset_name": args.dataset,
        "model_name": args.model,
        "reranker_type": reranker_type,
        "total_samples": total_samples,
        "evaluation_time_seconds": total_time,
        "average_time_per_sample": total_time / total_samples
    }

    for k in [1, 3, 5]:
        metrics[f"top_{k}_accuracy"] = correct[k] / total_samples

    # Save logs
    print(f"Saving logs to {logs_filename}")
    with open(logs_filename, "w") as f:
        for log in all_logs:
            f.write(json.dumps({**log, "dataset_name": args.dataset}) + "\n")

    # Save metrics
    print(f"Saving metrics to {metrics_filename}")
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    print("\nEvaluation Results:")
    print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")

    return metrics

if __name__ == "__main__":
    """
    hotpotqa:  query: str, positive: List[str], negative: List[str]
        All documents in positive are relevant to the query, while those in negative are not.
        When testing the re-ranker, both positive and negative documents have to be re-ranked,
        and then calculate top-k accuracy using positive documents (so maintain their label ids)
    """
    args = parse_args()
    evaluate_reranker(args)