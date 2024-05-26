import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from langchain.schema import HumanMessage, AIMessage

model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_rouge_1(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores['rouge1'].fmeasure

def calculate_conversational_perplexity(model, tokenizer, conversation, reference_text):
    device = next(model.parameters()).device 
    total_perplexity = 0
    num_responses = 0

    for message in conversation:
        if "User:" in message: 
            question = message.split("User:")[1].strip()
            u_q = f"Answer the following question: {question} using this report: {reference_text}"
            prompt_text = f'{{"role": "user", "content": "{u_q}"}}'
        elif "Counselor:" in message:
            counselor_response = message.split("Counselor:")[1].strip()        
            full_text = f'[{prompt_text}, {{"role": "assistant", "content": "{counselor_response}"}}]'           
            encodings = tokenizer(full_text, return_tensors='pt')
            input_ids = encodings.input_ids.to(device)
            labels = input_ids.clone()
            user_part_length = len(tokenizer(u_q)['input_ids'])
            labels[:, :user_part_length] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                log_likelihood = loss.item()
            
            perplexity = torch.exp(torch.tensor(log_likelihood)).item()
            total_perplexity += perplexity
            num_responses += 1

    average_perplexity = total_perplexity / num_responses if num_responses > 0 else float('inf')
    print(f"Total Perplexity: {total_perplexity}, Number of Responses: {num_responses}, Average Perplexity: {average_perplexity}")
    return average_perplexity
