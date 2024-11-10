import streamlit as st
import docx2txt
import torch
import numpy as np
from transformers import GPT2LMHeadModel, AutoTokenizer
import matplotlib.pyplot as plt

perplexity_score=0
burstiness_score=0

# Load Portuguese GPT-2 model and tokenizer directly
@st.cache_resource
def load_portuguese_gpt2_model():
    tokenizer = AutoTokenizer.from_pretrained('pierreguillou/gpt2-small-portuguese')
    model = GPT2LMHeadModel.from_pretrained('pierreguillou/gpt2-small-portuguese')
    return tokenizer, model

# Function to upload and read Word file
def upload_and_read_word(uploaded_file):
    text = docx2txt.process(uploaded_file)
    text = text.replace('\n', '')
    return text

# Function to calculate perplexity
def calculate_perplexity(text, tokenizer, model):
    tokens = tokenizer.encode(text, return_tensors='pt')
    max_length = 2048
    stride = 1024
    lls = []

    for i in range(0, tokens.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, tokens.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = tokens[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    perplexity = torch.exp(torch.stack(lls).sum() / end_loc)
    return perplexity.item()

# Function to calculate burstiness
def calculate_burstiness(text):
    sentences = text.split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
    if len(sentence_lengths) < 2:
        return 0
    mean_length = np.mean(sentence_lengths)
    variance_length = np.var(sentence_lengths)
    burstiness = variance_length / mean_length if mean_length != 0 else 0
    return burstiness

#Graphic Funcion
def plot_perplexity_burstiness(perplexity_score, burstiness_score):
    # Definindo limites para detecção
    perplexity_threshold = 35  # Limite sugerido para perplexity
    burstiness_threshold = 7  # Limite sugerido para burstiness
    
    # Determinando a cor com base nas métricas calculadas
    if perplexity_score > perplexity_threshold and burstiness_score > burstiness_threshold:
        color = 'blue'  # Provável de ser humano
    else:
        color = 'red'   # Provável de ser máquina
    
    # Criando o gráfico
    plt.figure(figsize=(8, 6))
    plt.scatter(perplexity_score, burstiness_score, color=color, s=100, label="Texto Analisado")
    
    # Adicionando uma anotação com os valores calculados das métricas, posicionada ao lado direito do ponto
    plt.annotate(f"Perplexity: {perplexity_score:.2f}\nBurstiness: {burstiness_score:.2f}", 
                 (perplexity_score, burstiness_score), 
                 textcoords="offset points", xytext=(15, 0), ha='left', va='center')
    
    # Adicionando linhas de limite com cores diferentes e rótulos com os valores
    plt.axvline(perplexity_threshold, color='blue', linestyle='--', 
                label=f"Limite de Perplexity: {perplexity_threshold}")
    plt.axhline(burstiness_threshold, color='green', linestyle='--', 
                label=f"Limite de Burstiness: {burstiness_threshold}")
    
    # Configurando o gráfico
    plt.xlabel("Perplexity")
    plt.ylabel("Burstiness")
    plt.title("Classificação de Texto com Base em Perplexity e Burstiness")
    
    # Posicionando a legenda abaixo do gráfico
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)
    
    # Exibindo o gráfico no Streamlit
    st.pyplot(plt)








# Streamlit Interface
st.image("NIDLogo.jpg")
st.title("NID GPT: AI Text Detection in Portuguese")

# Load the Portuguese GPT-2 model and tokenizer
try:
    tokenizer, model = load_portuguese_gpt2_model()
    st.success("Portuguese GPT-2 model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Upload file section
uploaded_file = st.file_uploader("Upload a Word document (.docx)", type=["docx"])

if uploaded_file is not None:
    # Read and preprocess text
    text = upload_and_read_word(uploaded_file)
    st.subheader("Uploaded Text Content:")
    #st.write(text)

    # Calculate perplexity and burstiness
    st.subheader("Calculating Perplexity and Burstiness...")
    with st.spinner('Calculating...'):
        perplexity_score = calculate_perplexity(text, tokenizer, model)
        burstiness_score = calculate_burstiness(text)

    # Display scores
    st.write(f"**Perplexity Score:** {perplexity_score}")
    st.write(f"**Burstiness Score:** {burstiness_score}")

    # Adjusted thresholds for classification
    perplexity_threshold = 35  # Revised threshold based on calibration suggestions
    burstiness_threshold = 7  # Revised burstiness threshold for further testing

    # Classification based on adjusted thresholds
    if perplexity_score < perplexity_threshold and burstiness_score > burstiness_threshold:
        st.success("The text is likely written by a **machine**.")
        plot_perplexity_burstiness(perplexity_score,burstiness_score)
    else:
        st.warning("The text is likely generated by a **human**.")
        plot_perplexity_burstiness(perplexity_score,burstiness_score)

else:
    st.info("Please upload a .docx file to analyze.")



  

