import os
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'

# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# 加载Documents
base_dir = 'OneFlower' # 文档的存放目录
documents = []
for file in os.listdir(base_dir): 
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'): 
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())


 # 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)       

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=OpenAIEmbeddings(), # 用OpenAI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain.chat_models import ChatOpenAI # ChatOpenAI模型
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个大模型工具 - OpenAI的GPT-3.5
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)

# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template
app = Flask(__name__) # Flask APP

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # 接收用户输入作为问题
        question = request.form.get('question')        
        
        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})
        
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)



"""
The code won't work if the data size is on TB Level. The followings are why it doesn't work:

| Step             | Problem                                                                   |
| ---------------- | ------------------------------------------------------------------------- |
| **1. Load**      | Loads **all documents into memory** at once — crashes at large scale      |
| **2. Split**     | Splits all documents **at once** — uses too much RAM                      |
| **3. Store**     | Uses in-memory `Qdrant` (`location=":memory:"`) — temporary, non-scalable |
| **4. Retrieval** | Uses `MultiQueryRetriever` — costly, slow on large corpus                 |
| **5. UI**        | No error handling, no context source display, no latency control          |


Here are the details:

For Step 1：Load

✅ Key Strategies for Large-Scale Document Loading
1. Process files in batches
Modify the loop to process one file (or a small batch) at a time: load → split → embed → store → release memory.

2. Use persistent vector database (not :memory:)
Use Qdrant, Chroma, or Weaviate running as a service (Docker or cloud) so that embeddings are stored to disk rather than memory.

3. Avoid keeping all raw documents/chunks in RAM
Split and embed as you go, and avoid collecting everything in a list like documents = [].

Additional Optimizations for TB-scale Data

| Technique                   | Description                                                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Parallel processing**     | Use `multiprocessing` or `concurrent.futures` to load/process multiple files concurrently.                                    |
| **Streaming loaders**       | If documents are very large individually (e.g., long PDFs), use stream-based loaders or chunk inside the loader.              |
| **Disk-backed storage**     | Use `LangChain + Qdrant/Chroma` running with persistent storage on disk.                                                      |
| **Checkpointing**           | Track which files are already processed (e.g., via filenames in a `.jsonl` log) to resume interrupted runs.                   |
| **Cloud storage pipelines** | For TB-scale data in the cloud, use S3 + Lambda or Spark + distributed embedding (e.g., HuggingFace + Faiss or Qdrant cloud). |



For Step 2:Split

✅ What to do instead: Split as you load
You should split each document immediately after loading, and then either:

store the chunks in the vector database right away (ideal), or

yield them one-by-one (streaming for further processing)

| Step                         | Memory Efficient? | Why                                                      |
| ---------------------------- | ----------------- | -------------------------------------------------------- |
| Load → Split → Store         | ✅ Yes             | Each file is handled individually and discarded from RAM |
| Load all → Split all → Store | ❌ No              | Everything stays in memory and scales poorly             |


For Step 3: Store

❌ Why it fails at scale:
from_documents() expects all documents at once in memory

:memory: creates a temporary database in RAM → wiped every restart

Not efficient for TB-scale ingestion

✅ Scalable Version of Step 3: Incremental Add with Persistent Vector DB
Example with Persistent Qdrant:
Start Qdrant locally or via Docker:

docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant


Then connect and add documents incrementally:

from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings

# Connect to persistent Qdrant
embedding_model = OpenAIEmbeddings()
vectorstore = Qdrant(
    collection_name="my_documents",
    embedding=embedding_model,
    url="http://localhost:6333",  # Qdrant server address
)

# In your loop:
chunks = text_splitter.split_documents(docs)
vectorstore.add_documents(chunks)  # ✅ Efficient, scalable


✅ Best Practices for Step 3 at Scale

| Practice                                          | Why                          |
| ------------------------------------------------- | ---------------------------- |
| Use `add_documents()` in small batches            | Keeps memory usage low       |
| Use persistent storage (Qdrant, Chroma, Weaviate) | Avoids losing data on crash  |
| Add `vectorstore.persist()` if needed             | Saves progress (Chroma)      |
| Optionally track processed files                  | Avoid re-processing on rerun |

For Step 4: Retrieval

For large-scale data, Step 4 needs careful handling too — especially around:

1.retriever configuration (efficiency, relevance)

2.query rewriting (e.g., MultiQueryRetriever)

3.batch vs. real-time mode

4.latency and memory use



| Aspect     | Recommendation                                            |
| ---------- | --------------------------------------------------------- |
| Retriever  | Use `.as_retriever(search_type="mmr")`                    |
| Chunks     | Use ~200–300 token chunks, retrieve top 3–5              |
| MultiQuery | Disable unless precision is critical                      |
| LLM        | Use `streaming=True` for fast UI                          |
| Memory     | Avoid loading all documents into prompt at once           |
| Cost       | Watch OpenAI token usage when retrieving too much context |


For Step 5 UI:

| Feature          | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| 🔐 Rate limiting | Use Flask extensions or API gateways to limit abuse        |
| 🧠 Caching       | Store recent queries & results to reduce repeated LLM cost |
| 💾 Logging       | Store Q\&A pairs, timestamps, etc. for analysis/debugging  |
| 🌐 Frontend      | Use Streamlit, Gradio, or React for a more responsive UI   |

| Concern        | What to Do                                   |
| -------------- | -------------------------------------------- |
| Latency        | Use spinners, async, streaming               |
| Debuggability  | Show sources (retrieved docs)                |
| Robustness     | Add error handling, token limits             |
| UI flexibility | Consider Streamlit or Gradio for modern feel |


✅ Updated Version of the Code (Step 1–5)

import os
from flask import Flask, request, render_template
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ✅ Step 0: Set OpenAI API key (secure way recommended)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-fallback-key")  # or use dotenv

# ✅ Step 1–3: Efficient document processing and vector storage
base_dir = 'OneFlower'  # Folder containing many files
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
embedding_model = OpenAIEmbeddings()

# ✅ Use persistent Qdrant
vectorstore = Qdrant(
    collection_name="my_documents",
    embedding=embedding_model,
    url="http://localhost:6333"  # run Qdrant via Docker or Cloud
)

# ✅ Efficient batch processing
for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)

    # Choose loader by extension
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        continue

    try:
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        vectorstore.add_documents(chunks)  # ✅ Add to persistent DB
    except Exception as e:
        print(f"Failed to process {file}: {e}")

# ✅ Step 4: Setup scalable retriever and QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=False)

retriever = vectorstore.as_retriever(
    search_type="mmr",  # better diversity on large datasets
    search_kwargs={"k": 5, "fetch_k": 15}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # So we can show sources
)

# ✅ Step 5: Web UI
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form.get('question')
        try:
            result = qa_chain({"query": question})
            return render_template('index.html',
                                   question=question,
                                   answer=result["result"],
                                   sources=result.get("source_documents", []))
        except Exception as e:
            return render_template('index.html',
                                   question=question,
                                   answer="Error: " + str(e),
                                   sources=[])
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)


"""