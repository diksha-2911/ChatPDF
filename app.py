from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
import os
# from langchain_core.output_parsers import StrOutputParser

# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# from langchain_community.llms import HuggingFaceHub


def main():
    

    
    load_dotenv() 
    # print("Hello World")
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with your PDF 💬")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    # #upload
    pdf = st.file_uploader("Upload a PDF", type="pdf")

    #extract
    if pdf is not None:
        st.session_state.uploaded_file = pdf
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #split into chunks using langchain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
        )
        chunks = text_splitter.split_text(text)

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
        os.environ["HF_TOKEN"] = "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"
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'pcsk_3BPfoR_7jHZ1KAFgjX4RQba36F7LJfqERr7mXf2HuLaCF5EXjMYWkuoMnP9pXYbw2LuhT1')
        
        #create embeddings
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

        pc = Pinecone(api_key="PINECONE_API_KEY")

        index_name = "langchainpinecone"

        docsearch = PineconeVectorStore.from_texts([chunk for chunk in chunks], embeddings, index_name=index_name)


        # llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        # chain=load_qa_chain(llm, chain_type="stuff")

        # prompt = ChatPromptTemplate.from_messages([
        #     ("ai", "Answer user's question"),
        #     ("human", "Question : {question}"),
        # ])

        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            max_new_tokens= 512,
            temperature=0.5,
            huggingfacehub_api_token="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",
        )
        # output = StrOutputParser()
        # chain = prompt | llm | output

        chain=load_qa_chain(llm, chain_type="stuff")
        

        #user input
        user_question = st.text_input("Ask question to the PDF")
        if user_question:
            docs = docsearch.similarity_search(user_question, k=1)
            # prompt = PromptTemplate.from_template(template)
            # chain = create_stuff_documents_chain(llm, prompt)
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)


        

if __name__ == "__main__":
    main()