from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable
from langchain_core.prompt_values import StringPromptValue
from langchain.schema import HumanMessage, AIMessage
import chainlit as cl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import pandas as pd
import os
import time
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from disease_extractorLLama import generate_counselor_report, read_file
from Evaluations import calculate_conversational_perplexity, model, tokenizer, calculate_rouge_1

modalname = "LLama2"
pdf_path = "MedicalReports"
file_name= "medicalreport.pdf"
title, script = generate_counselor_report(pdf_path, file_name)
torch.random.manual_seed(0)

DB_FAISS_PATH = 'vectorstore/db_faiss'
nlp = spacy.load("en_core_web_sm")

def calculate_cosine_similarity(predicted, reference):
    vectorizer = CountVectorizer().fit_transform([predicted, reference])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def calculate_jaccard_similarity(predicted, reference):
    pred_set = set(predicted.split())
    ref_set = set(reference.split())
    intersection = pred_set.intersection(ref_set)
    union = pred_set.union(ref_set)
    jaccard_sim = len(intersection) / len(union)
    return jaccard_sim

def log_similarities(modalname, conversation_id, cosine_similarities, average_cosine, jaccard_similarities, average_jaccard):
    file_path = modalname +"_similarity_results.csv"
    columns = ['conversation_id', 'cosine_similarities', 'average_cosine', 'jaccard_similarities', 'average_jaccard']
    data = {
        'conversation_id': conversation_id,
        'cosine_similarities': cosine_similarities,
        'average_cosine': average_cosine,
        'jaccard_similarities': jaccard_similarities,
        'average_jaccard': average_jaccard
    }
    
    if not os.path.exists(file_path):
        df = pd.DataFrame([data], columns=columns)
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(file_path, index=False)

def log_perplexity(modalname, conversation_id, perplexity):
    file_path = modalname+"_perplexity_results.csv"
    columns = ['conversation_id', 'perplexity']
    data = {'conversation_id': conversation_id, 'perplexity': perplexity}
    
    if not os.path.exists(file_path):
        df = pd.DataFrame([data], columns=columns)
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(file_path, index=False)

def log_rouge(modalname, conversation_id, rouge_1_scores, average_rouge_1):
    file_path = modalname+"_rouge_results.csv"
    columns = ['conversation_id', 'rouge_1_scores', 'average_rouge_1']
    data = {
        'conversation_id': conversation_id,
        'rouge_1_scores': rouge_1_scores,
        'average_rouge_1': average_rouge_1
    }
    
    if not os.path.exists(file_path):
        df = pd.DataFrame([data], columns=columns)
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(file_path, index=False)

def set_custom_prompt(script):
    """
    Prompt template for QA retrieval for each vectorstore, including the script.
    """
    custom_prompt_template = f"""
    You are a professional genetic counsellor and a doctor. Use the provided information to have a conversation with a patient.
    Make sure the conversation is genetic counsellor-directed. The genetic counsellor should control the conversation.
    Always remember the context of the previous response. Show affection and concern to the patient and always end the response with a question for the patient please.

    Initial Script: {script}

    Context: {{context}}
    Question: {{question}}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

class HFLLM(Runnable):
    def __init__(self, pipe, generation_args):
        self.pipe = pipe
        self.generation_args = generation_args

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, StringPromptValue):
            input_text = input.text
        else:
            input_text = input
        messages = [{"role": "user", "content": input_text}]
        output = self.pipe(messages, **self.generation_args)
        assistant_response = None
        for message in output:
            if message["role"] == "assistant":
                assistant_response = message["content"]
                break
    
        return assistant_response

    def run(self, input):
        return self.invoke(input)

def load_llm():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", 
            device_map="auto",  # Automatically distribute layers across devices
            torch_dtype=torch.float16,  # Use float16 for lower memory consumption
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        pipe = pipeline(
            "conversational",
            model=model,
            tokenizer=tokenizer,
        )
        generation_args = {
            "max_new_tokens": 500,
            "do_sample": False
        }
        hf_llm = HFLLM(pipe=pipe, generation_args=generation_args)
        return hf_llm
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory: {e}")
        torch.cuda.empty_cache()
        return None

def retrieval_qa_chain(llm, prompt, db, memory):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt},
                                           memory=memory,
                                           output_key="result"
                                           )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    hf_llm = load_llm()
    # title, script = generate_counselor_report(pdf_path, file_name)
    qa_prompt = set_custom_prompt(script)
    memory = ConversationBufferMemory(llm=hf_llm, max_token_limit=512, memory_key="chat_history", output_key="result")
    qa = retrieval_qa_chain(hf_llm, qa_prompt, db, memory)
    return qa, memory

def final_result(query):
    import pdb
    pdb.set_trace()
    qa, memory = qa_bot()
    response = qa({'query': query})
    conversation_history = memory.chat_memory.messages
    return response

@cl.on_chat_start
async def start():
    chain, memory = qa_bot()
    msg = cl.Message(content="Getting your Genetic Counsellor ready!")
    await msg.send()
    msg.content = "Hi, " + script + "\nHow may I help you today with this information?"
    await msg.update()
    cl.user_session.set("chain", chain)
    cl.user_session.set("memory", memory)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")

    if any(thank_word in message.content.lower() for thank_word in ["thank you", "thanks"]):
        await cl.Message(content="You're welcome! If you have any more questions or need further assistance, feel free to reach out.").send()
    else:
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        
        cb.answer_reached = True
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res.get("source_documents", [])
        if sources:
            source_details = ", ".join([f"{doc.metadata['source']} page {doc.metadata['page']}" for doc in sources])[:150]
            answer = f"{answer}\n\n**SOURCES:**\n{source_details}"
        else:
            answer += "\nNo sources found"
        await cl.Message(content=answer).send()
        conversation_history = memory.chat_memory.messages
        conversation = []
        predicted_responses = []
        for msg in conversation_history:
            # Replace with actual attribute access based on message structure
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                conversation.append(f"{msg.role.capitalize()}: {msg.content}")
            else:
                if isinstance(msg, HumanMessage):
                    conversation.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    conversation.append(f"Counselor: {msg.content}")
                    predicted_responses.append(msg.content)
            perplexity = calculate_conversational_perplexity(model, tokenizer, conversation)
            conversation_id = f"conv_{int(time.time())}"
            log_perplexity(modalname, conversation_id, perplexity)
            reference_path= "ReferenceReport/example.docx.pdf"
            reference_text = read_file(reference_path)
            rouge_1_scores = [calculate_rouge_1(predicted, reference_text) for predicted in predicted_responses]
            average_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0
            log_rouge(modalname, conversation_id, rouge_1_scores, average_rouge_1)
            
                    # Calculate Cosine and Jaccard similarities
        cosine_similarities = [calculate_cosine_similarity(predicted, reference_text) for predicted in predicted_responses]
        average_cosine = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0
        jaccard_similarities = [calculate_jaccard_similarity(predicted, reference_text) for predicted in predicted_responses]
        average_jaccard = sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0

        # Log similarities
        log_similarities(modalname, conversation_id, cosine_similarities, average_cosine, jaccard_similarities, average_jaccard)

        print(f"Cosine Similarities: {cosine_similarities}")
        print(f"Average Cosine Similarity: {average_cosine}")
        print(f"Jaccard Similarities: {jaccard_similarities}")
        print(f"Average Jaccard Similarity: {average_jaccard}")
