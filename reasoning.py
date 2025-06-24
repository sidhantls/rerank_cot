from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import time

prefix = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. First, provide a brief reasoning about the similarity between the query and document, and whether the query can be answered by the document. Then provide your final answer as "yes" or "no".

Examples:

Query: "What is the population of Tokyo?"
Document: "Tokyo is the capital city of Japan with a population of approximately 14 million people in the metropolitan area."
Reasoning: The query asks specifically for Tokyo's population, and the document directly provides this information stating "approximately 14 million people." The document perfectly matches the query's information need.
Final Answer: yes

Query: "How to cook pasta?"
Document: "The history of Rome dates back to ancient times when it was founded in 753 BC."
Reasoning: The query is asking for cooking instructions for pasta, but the document discusses the historical founding of Rome. There is no connection between the query and document content - they cover completely different topics.
Final Answer: no

Query: "What causes earthquakes?"
Document: "Earthquakes are caused by the sudden release of energy stored in rocks beneath the Earth's surface, typically along fault lines where tectonic plates meet."
Reasoning: The query asks about earthquake causes, and the document provides a clear scientific explanation about energy release in rocks and tectonic plate movement. This directly answers the query with relevant geological information.
Final Answer: yes<|im_end|>
<|im_start|>user
"""

# Improved prompt with reasoning examples and instructions
prefix_v2 = """<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. First, provide a detailed reasoning about the similarity between the query and document, analyzing both semantic overlap and information completeness. Then reflect on your reasoning to ensure accuracy. Finally, provide your answer as \"yes\" or \"no\".

Examples:

Query: "What are the health benefits of drinking green tea?"
Document: "Green tea contains high levels of antioxidants called catechins, particularly EGCG, which may help reduce inflammation, boost metabolism, and lower the risk of heart disease and certain cancers."
Reasoning: The query seeks information about health benefits of green tea, and the document provides specific health benefits including antioxidant content, anti-inflammatory properties, metabolic effects, and disease prevention. The semantic overlap is very high as both focus on green tea's health aspects. The document comprehensively addresses the query's intent. Upon reflection, this is a clear match - the document directly answers what the query is asking for with specific, relevant health benefits.
Final Answer: yes

Query: "How to fix a leaky faucet?"
Document: "A leaky faucet typically occurs due to worn-out washers, O-rings, or valve seats. To fix it, turn off the water supply, disassemble the faucet handle, replace the damaged parts, and reassemble."
Reasoning: The query asks for repair instructions for a leaky faucet, and the document provides both the causes and step-by-step repair process. There's complete semantic alignment - both are about faucet repair. The document offers practical, actionable information that directly addresses the query. Reflecting on this analysis, the document fully satisfies the information need with both diagnostic and procedural guidance.
Final Answer: yes

Query: "What is machine learning?"
Document: "Artificial intelligence has revolutionized many industries in recent years, with applications ranging from autonomous vehicles to medical diagnosis systems."
Reasoning: The query asks for a definition or explanation of machine learning, while the document discusses AI applications in various industries. Although machine learning is a subset of AI, there's limited semantic overlap - the document doesn't define or explain machine learning specifically. The document talks about AI broadly but doesn't address the core question about what machine learning is.
Final Answer: no

Query: "Best restaurants in Paris"
Document: "The weather in Tokyo today is sunny with a temperature of 25 degrees Celsius and light winds from the east."
Reasoning: The query seeks restaurant recommendations in Paris, while the document provides weather information for Tokyo. These topics are completely unrelated - different cities, different information types (dining vs. weather). There's no semantic overlap whatsoever. The document cannot help answer the query in any meaningful way. Reflecting on this analysis, this is clearly a complete mismatch with no relevance between query and document content.
Final Answer: no<|im_end|>
<|im_start|>user\n"""

def get_token_ids(tokenizer):
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    return token_false_id, token_true_id

def build_prefix_and_suffix(tokenizer, use_v2_prefix=False):
    prefix_to_use = prefix_v2 if use_v2_prefix else prefix
    suffix = "<|im_end|>\n<|im_start|>assistant\nReasoning: "
    prefix_tokens = tokenizer.encode(prefix_to_use, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    return prefix_to_use, suffix, prefix_tokens, suffix_tokens

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output

def process_inputs(pairs, tokenizer, model, prefix_tokens, suffix_tokens, max_length):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits_traditional(inputs, model, token_true_id, token_false_id, batch_size=8):
    input_ids = inputs['input_ids']
    # If attention_mask exists, use it
    attention_mask = inputs.get('attention_mask', None)
    num_samples = input_ids.size(0)
    scores = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = {'input_ids': input_ids[start:end]}
        if attention_mask is not None:
            batch['attention_mask'] = attention_mask[start:end]
        batch_scores = model(**batch).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores.extend(batch_scores[:, 1].exp().tolist())
    return scores

@torch.no_grad()
def compute_scores_with_reasoning(inputs, model, tokenizer, token_true_id, token_false_id, batch_size=8):
    reasoning_suffix = "<|im_end|>\n<|im_start|>assistant\nReasoning: "
    reasoning_suffix_tokens = tokenizer.encode(reasoning_suffix, add_special_tokens=False)

    reasoning_inputs = {}
    for key in inputs:
        reasoning_inputs[key] = inputs[key].clone()

    input_ids = reasoning_inputs['input_ids']
    attention_mask = reasoning_inputs.get('attention_mask', None)
    num_samples = input_ids.size(0)
    reasoning_outputs = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = {'input_ids': input_ids[start:end]}
        if attention_mask is not None:
            batch['attention_mask'] = attention_mask[start:end]
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch,
                max_new_tokens=70,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values = DynamicCache()
            )
        reasoning_outputs.extend(batch_outputs)

    final_answer_prefix = "\n\nFinal Answer: "
    final_inputs = []
    reasonings = []

    for i, reasoning_output in enumerate(reasoning_outputs):
        generated_reasoning = tokenizer.decode(
            reasoning_output[len(inputs['input_ids'][i]):],
            skip_special_tokens=True
        )
        reasoning_text = generated_reasoning.replace("Reasoning: ", "").strip()
        if "Final Answer:" in reasoning_text:
            reasoning_text = reasoning_text.split("Final Answer:")[0].strip()
        reasonings.append(reasoning_text)
        full_text = tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
        full_text += reasoning_text + final_answer_prefix
        final_input = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
        final_inputs.append(final_input.squeeze(0))

    max_len = max(len(inp) for inp in final_inputs)
    padded_inputs = []
    attention_masks = []
    for inp in final_inputs:
        pad_length = max_len - len(inp)
        if pad_length > 0:
            padded_inp = torch.cat([torch.full((pad_length,), tokenizer.pad_token_id, dtype=inp.dtype), inp])
            attention_mask = torch.cat([torch.zeros(pad_length, dtype=inp.dtype), torch.ones(len(inp), dtype=inp.dtype)])
        else:
            padded_inp = inp
            attention_mask = torch.ones(len(inp), dtype=inp.dtype)
        padded_inputs.append(padded_inp)
        attention_masks.append(attention_mask)

    batch_inputs = {
        'input_ids': torch.stack(padded_inputs).to(model.device),
        'attention_mask': torch.stack(attention_masks).to(model.device)
    }

    with torch.no_grad():
        for start in range(0, batch_inputs['input_ids'].size(0), batch_size):
            end = min(start + batch_size, batch_inputs['input_ids'].size(0))
            batch = {
                'input_ids': batch_inputs['input_ids'][start:end],
                'attention_mask': batch_inputs['attention_mask'][start:end]
            }
            outputs = model(**batch)
            logits = outputs.logits[:, -1, :]
            yes_logits = logits[:, token_true_id]
            no_logits = logits[:, token_false_id]
            yes_no_logits = torch.stack([no_logits, yes_logits], dim=1)
            probabilities = torch.nn.functional.softmax(yes_no_logits, dim=1)
            if start == 0:
                yes_probabilities = probabilities[:, 1].cpu().tolist()
            else:
                yes_probabilities.extend(probabilities[:, 1].cpu().tolist())

    return yes_probabilities, reasonings

@torch.no_grad()
def compute_scores_with_reasoning_v2(query, inputs, model, tokenizer, documents, max_length=512, batch_size=8):
    # Create empty logs file

    reasoning_suffix = "<|im_end|>\n<|im_start|>assistant\nReasoning: "
    reasoning_suffix_tokens = tokenizer.encode(reasoning_suffix, add_special_tokens=False)

    reasoning_inputs = {}
    for key in inputs:
        reasoning_inputs[key] = inputs[key].clone()

    input_ids = reasoning_inputs['input_ids']
    attention_mask = reasoning_inputs.get('attention_mask', None)
    num_samples = input_ids.size(0)
    reasoning_outputs = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = {'input_ids': input_ids[start:end]}
        if attention_mask is not None:
            batch['attention_mask'] = attention_mask[start:end]
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        reasoning_outputs.extend(batch_outputs)

    reasoning_text = []
    for i, reasoning_output in enumerate(reasoning_outputs):
        generated_reasoning = tokenizer.decode(
            reasoning_output[len(inputs['input_ids'][i]):],
            skip_special_tokens=True
        )

        generated_reasoning = generated_reasoning.split("Final Answer:")[0].strip()
        documents[i] = documents[i] + " Reasoning: " + generated_reasoning + "\n"
        reasoning_text.append(generated_reasoning)

    # Append first 5 documents with their reasoning to logs.txt
    with open('logs.txt', 'a') as f:
        for i, doc in enumerate(documents[:2]):
            f.write(f"Document {i+1}:\n{doc}\n\n")

    yes_probabilities = predict(query, documents, model, tokenizer, max_length=512, batch_size=8)

    return yes_probabilities, reasoning_text

import baseline
def predict(query, documents, model, tokenizer, max_length=512, batch_size=8):
    def get_special_tokens(tokenizer):
        # Modified to include reasoning
        # prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query, Reasoning, and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        # default prefix
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        return prefix_tokens, suffix_tokens

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    prefix_tokens, suffix_tokens = baseline.get_special_tokens(tokenizer)
    token_false_id, token_true_id = baseline.get_token_ids(tokenizer)
    scores = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        pairs = [baseline.format_instruction(task, query, doc) for doc in batch_docs]
        inputs = baseline.process_inputs(pairs, tokenizer, max_length, prefix_tokens, suffix_tokens, model)
        batch_scores = baseline.compute_logits(inputs, model, token_true_id, token_false_id)
        scores.extend(batch_scores)
    return scores

def predict_with_reasoning(query, documents, model, tokenizer, use_v2_prefix=False, batch_size=8, max_length=512*2):
    token_false_id, token_true_id = get_token_ids(tokenizer)
    prefix, suffix, prefix_tokens, suffix_tokens = build_prefix_and_suffix(tokenizer, use_v2_prefix=use_v2_prefix)
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [format_instruction(task, query, doc) for doc in documents]
    inputs = process_inputs(pairs, tokenizer, model, prefix_tokens, suffix_tokens, max_length=max_length)
    # For compute_logits_traditional, pass batch_size as argument
    # prob_scores = compute_logits_traditional(inputs, model, token_true_id, token_false_id, batch_size=batch_size)
    # prob_scores, reasonings = compute_scores_with_reasoning(inputs, model, tokenizer, token_true_id, token_false_id, batch_size=batch_size)
    prob_scores, reasonings = compute_scores_with_reasoning_v2(query, inputs, model, tokenizer, documents, max_length=max_length, batch_size=batch_size)
    return prob_scores, reasonings

if __name__ == "__main__":
    # Example usage
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    query = "What is the capital of China?"
    documents = [
        "Beijing is the capital of China.",
        "The Great Wall of China is a famous landmark.",
        "China has a rich history and culture."
    ]

    start_time = time.time()
    prob_scores, reasonings = predict_with_reasoning(query, documents, model, tokenizer)
    end_time = time.time()
    print("Probabilities with Reasoning:", prob_scores)
    print("Reasonings:", reasonings)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
