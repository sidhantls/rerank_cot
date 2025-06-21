import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name="Qwen/Qwen3-Reranker-8B"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    return model, tokenizer, device

def get_special_tokens(tokenizer):
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    return prefix_tokens, suffix_tokens

def get_token_ids(tokenizer):
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    return token_false_id, token_true_id

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output

def process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens, model):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    device = model.device if hasattr(model, 'device') else torch.device('cpu')
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, model, token_true_id, token_false_id):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def predict(query, documents, model, tokenizer, max_length=512):
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [format_instruction(task, query, doc) for doc in documents]
    prefix_tokens, suffix_tokens = get_special_tokens(tokenizer)
    token_false_id, token_true_id = get_token_ids(tokenizer)
    inputs = process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens, model)
    scores = compute_logits(inputs, model, token_true_id, token_false_id)
    return scores


if __name__ == "__main__":
    # Requires transformers>=4.51.0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-Reranker-0.6B"
    model, tokenizer, device = load_model_and_tokenizer(model_name)

    queries = [
        "What is the capital of China?",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    for query in queries:
        scores = predict(query, documents, model, tokenizer)
        for i, doc in enumerate(documents):
            print(f"Query: {query}")
            print(f"Document: {doc}")
            print(f"Score: {scores[i]:.4f}")
            print("-" * 40)
