# Genetic Counsellor - Master Thesis:
This Project is in collaboration with [Arkus-AI](https://arkus.ai/) which works in the area of genetic counseling. The main aim of this project was to explore the area of genetics combined with LLMs and see how can we integrating the LLM models as a Genetic Counsellor
and observe how well they work in regards to providing assistance to the patients and helping in answering their queries.

# Methodologies Used:
 - Document-based Generations
 -  Retrieval-Augmented Generation (RAG)
 - LLMChains

### Tools:
- Python
- Langchain
- LLMChains
- LLaMA2
- Phi-3
- HuggingFace
- Transformers
- SentenceTransformers

Note: To be able to run the project, you might need to have a ```HuggingFace API``` and also you must have access granted to use the ```LLaMA2``` model by Facebook/Meta. To get the permission, please visit this website and fill in the form to get permission 
      at this [URL](https://huggingface.co/meta-llama/Llama-2-7b-hf)

### Use of Models:
- [LLaMA2 model](https://github.com/ZeeshanM96)
- [Phi-3 model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

### Model Specifications:
- LLaMA2 model (meta-llama/Llama-2-7b-hf):
  - Architecture: Transformer-based
  - Token Size: Up to 2 trillion tokens
  - Model Sizes: Available in different sizes (7B, 13B, 30B parameters)
  - We made use of the model with size 7B.
  - Context window size 4K tokens

- Phi-3 model (Microsoft/Phi-3-mini-128k-instruct):
  - Architecture: Transformer-based.
  - Token Size: Trained over 3+ trillion tokens
  - Model Sizes: Configurations with 3.8B parameters
  - We made use of the model with the configuration of 3.8B
  - Context window size 128K tokens
  
### Requirements:
This project doesn't require much of the system requirements, but it does require a good GPU to run the models smoothly since the LLaMA model requires higher computational power, so in order to run your model smoothly make sure you have enough memory available.

### How to Run:
Please follow the instructions to run the application.
#### Activate the virtual Env
```source venv/bin/activate```

#### Install the dependencies
```pip install -r requirements.txt```

#### Install these dependencies separately
```
pip install pypdf
pip install spacy
python -m spacy download en_core_web_sm
pip install sentence-transformers
pip install faiss-gpu
pip install chainlit
pip install pandas
pip install PyMuPDF
pip install accelerate
pip install wikipedia
pip install rouge-score
```

#### This needs to be installed only once
```python -m spacy download en_core_web_sm```

#### If needed install below:
```pip install -U langchain-community```

#### Run the ingest file to create the embeddings:
```python ingest.py```

#### Run the model script:
- For LLaMA2:
```chainlit run model.py -w```
- For Phi-3:
```chainlit run model1.py -w```

- if needed install Ctransformers package and run the model script again:
```pip install ctransformers```

- To check the usage of GPU
```nvidia-smi```

- To resolve port already in use issue, check the p-ids using that port number:
```lsof -i :8000```

- Stop the processes using the 8000 with the p-ids:
```kill -9 <PID>```

## Architecture Diagram:
![ArchitecturalDiagram2](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/76798278-9b90-4b47-bec2-cdbf16ade550)

## Results & Evaluations:
- Perplexity Scores:
![image](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/29ec976a-e060-4f68-97c1-d2de09d160f0)

- Rouge-1 Scores:
![image](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/b7068f09-6dd0-4341-a833-83f632b06992)

- Cosine & Jaccard Similarity Scores:
![image](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/c775353a-25ed-4256-9139-0a5857f467bb)


## Discussion:
Results demonstrate that LLaMA2, with its larger parameter count, consistently outperforms the smaller Phi-3 model. The larger model’s ability to better capture and understand complex contexts likely contributes to its superior performance. These findings
underscore the importance of considering model size when implementing document grounding methodologies, as larger models may provide more accurate and reliable results.

## Test Samples:

![Phi-3 - Sample Covo-1](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/d39fc7d4-0122-4a58-8e0d-bc3c6243a5d0)
![Phi-3 - Sample Covo-2](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/02ad21b4-7214-4d3a-8cea-99b9eee3b964)
![Phi-3 - Sample Covo-3](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/b50edb71-2418-4a7f-805a-81db27ee44c4)
![Phi-3 - Sample Covo-4](https://github.com/ZeeshanM96/GeneticCounsellor/assets/116648836/ea802023-05d3-47cb-a303-cb888b703987)

## Acknowledgements:
This project was done with the support of my supervisor: [Alessio Galatolo](https://www.alessiogalatolo.com/)


