import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.runnables import Runnable
import torch
from langchain_core.prompt_values import StringPromptValue
import fitz  # PyMuPDF

model_name = "Llama-2"
path = "graphs"

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",  
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
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

hf_llm = HFLLM(pipe=pipe, generation_args=generation_args)

def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_file(file_path):
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    else:
        raise ValueError("Unsupported file format")

def generate_counselor_report(directory, file_name):
    file_path = os.path.join(directory, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    prompt = read_file(file_path)

    title_template = PromptTemplate(
        input_variables=['topic'],
        template='Extract the cancer gene variants from the text: {topic} and briefly explain them using Wikipedia research.'
    )

    script_template = PromptTemplate(
        input_variables=['title', 'wikipedia_research'],
        template='Extract cancer gene variants from the text and briefly explain each of them based on the given title: {title} while leveraging this Wikipedia research: {wikipedia_research}.'
    )
    
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    title_chain = LLMChain(llm=hf_llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=hf_llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
    wiki = WikipediaAPIWrapper()
    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    print(script)
    return title, script


directory = "MedicalReports"
file_name= "medicalreport.pdf"
generate_counselor_report(directory, file_name)