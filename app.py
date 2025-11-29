from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
CORS(app)

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = download_hugging_face_embeddings()

index_name = "medibuddy" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity")

# chatModel = ChatOpenAI(model="gpt-4o")
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# --- Hugging Face Medical Model (Better for medical questions) ---
hf_llm = HuggingFaceEndpoint(
    repo_id="Intelligent-Internet/II-Medical-8B",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.0,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# --- Alternative: Generic Zephyr model ---
# hf_llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=256,
#     temperature=0.1,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
# )

chatModel = ChatHuggingFace(llm=hf_llm)  
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        print("Received request")
        msg = request.form.get("msg", "")
        print(f"Message: {msg}")

        if not msg:
            return jsonify({"answer": "Please provide a message"}), 400

        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        if "StopIteration" in traceback.format_exc() or "provider" in traceback.format_exc().lower():
            error_msg = "HuggingFace API configuration error. Please check your HUGGINGFACEHUB_API_TOKEN in the .env file."
        return jsonify({"answer": f"⚠️ {error_msg}"}), 200


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
