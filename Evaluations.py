import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

model_name = "microsoft/Phi-3-mini-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_conversational_perplexity(model, tokenizer, conversation):
    conversation_text = ' '.join(conversation)
    encodings = tokenizer(conversation_text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        log_likelihood = loss.item()
    perplexity = torch.exp(torch.tensor(log_likelihood))
    return perplexity.item()

def calculate_rouge_1(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores['rouge1'].fmeasure
