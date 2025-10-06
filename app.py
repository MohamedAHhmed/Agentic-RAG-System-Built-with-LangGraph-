import os
from typing import TypedDict, Annotated
from uuid import uuid4
from functools import partial

from flask.cli import load_dotenv

import faiss
import gradio as gr
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage,
)

load_dotenv()

# ========= Model & Embeddings =========
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.02,
    max_tokens=4096,
    openai_api_base="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_llm = HuggingFaceEmbeddings(model_name=model_name)
embedding_dim = len(embedding_llm.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embedding_llm,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# ========= Functions =========
def add_documents_to_vdb(urls, vector_store, logs):
    try:
        logs.append("📥 Fetching documents...")
        docs = []
        for url in urls:
            logs.append(f"🔗 Loading URL: {url}")
            docs.extend(WebBaseLoader(url).load())

        logs.append(f"📄 Total documents loaded: {len(docs)}")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(docs)
        logs.append(f"✂️ Split into {len(doc_chunks)} chunks.")

        uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
        vector_store.add_documents(documents=doc_chunks, ids=uuids)
        logs.append("💾 Documents added to FAISS store.")

        dense_retriever = vector_store.as_retriever(k=5)
        bm25_retriever = BM25Retriever.from_documents(doc_chunks)
        bm25_retriever.k = 8

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.4, 0.6],
        )
        logs.append("✅ Hybrid retriever ready.")
        return hybrid_retriever
    except Exception as e:
        logs.append(f"❌ Error adding documents: {str(e)}")
        return False

# ========= Graph Nodes =========
class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    rewrite_count: int
    summary: str

def grade_documents(state):
    print("---CHECK RELEVANCE---")
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""You are a helpful assistant that checks if documents match user questions. 
                  Here is the document: \n\n {context} \n\n
                  Here is the user question: {question} \n
                  Does this document help answer the question? \n
                  Reply with 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if question is None:
        question = messages[0].content

    docs = last_message.content
    current_count = state.get("rewrite_count", 0)
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        return "generate"
    elif score == "no":
        if current_count >= 3:
            return "search_google"
        else:
            return "rewrite"

def agent(state, tools):
    print("---CALL AGENT---")
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        system_message = f"""Summary of conversation earlier: {summary} 
        NOTE: If the user asks about previous questions or conversation history, answer directly from this summary. For technical questions, Always use the retriever tool ."""
        messages = [SystemMessage(content=system_message)] + messages

    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    chat_model = chat_model.bind_tools(tools)
    response = chat_model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    print("---REWRITE---")
    messages = state["messages"]
    question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    new_count = state.get("rewrite_count", 0) + 1
    msg = [
        HumanMessage(
            content=f"""Here is the initial question:\n {question}\n
            Formulate an improved question:"""
        )
    ]
    response = model.invoke(msg)
    return {"messages": [response], "rewrite_count": new_count}

def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if question is None:
        question = messages[0].content

    docs = messages[-1].content
    prompt_template = hub.pull("rlm/rag-prompt")
    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    output_parser = StrOutputParser()
    rag_chain = prompt_template | chat_model | output_parser
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [AIMessage(content=response)]}

def search_google(state):
    print("---SEARCH GOOGLE---")
    messages = state["messages"]
    question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if question is None:
        question = messages[0].content
    tools = load_tools(["google-serper"], serper_api_key=os.getenv("SERPER_API_KEY"))
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    agent = create_react_agent(llm, tools)
    response = agent.invoke({"messages": [("human", question)]})
    final_answer = response["messages"][-1].content
    return {
        "messages": [AIMessage(content=final_answer)],
        "rewrite_count": 0
    }

def summarize_conversation(state):
    print("---SUMMARIZE---")
    summary = state.get("summary", "")
    if summary:
        summary_message = f"This is summary of the conversation to date: {summary}"
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
    return {"summary": response.content, "messages": delete_messages}

def should_summarize(state):
    if len(state["messages"]) > 10:
        return True
    return False


# ========= Build Graph =========
def build_graph(retriever_tool):
    workflow = StateGraph(CustomState)
    workflow.add_node("agent", partial(agent, tools=[retriever_tool]))
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("summarize", summarize_conversation)
    workflow.add_node("search_google", search_google)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {"rewrite": "rewrite", "generate": "generate", "search_google": "search_google"},
    )
    workflow.add_conditional_edges(
        "generate",
        should_summarize,
        {True: "summarize", False: END},
    )
    workflow.add_edge("summarize", END)
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("search_google", "generate")

    within_thread_memory = MemorySaver()
    return workflow.compile(checkpointer=within_thread_memory)


# ========= Global graph (memory-enabled) =========
global_graph = None

def setup_graph(links, logs):
    retriever = add_documents_to_vdb(links, vector_store, logs)
    if not retriever:
        return None
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return info from blog posts",
    )
    return build_graph(retriever_tool)

# ========= Chat function =========
def chat_fn(message, links, logs_state):
    logs = []
    links = [u.strip().strip('"').strip("'") for u in links.split(",") if u.strip()]
    logs.append(f"🔗 **Links used:** {links}")

    global global_graph
    if global_graph is None:  # build once
        global_graph = setup_graph(links, logs)
        logs.append("📊 **Graph compiled & memory enabled.**")

    inputs = {"messages": [HumanMessage(content=message)], "rewrite_count": 0}
    config = {"configurable": {"thread_id": "user-1"}}  # fixed thread
    logs.append(f"🤖 **User asked:** {message}")

    output = global_graph.invoke(inputs, config)
    answer = output["messages"][-1].content
    logs.append("✅ **Answer generated.**")

    logs_state += "\n".join(logs) + "\n\n"
    return answer, "", logs_state

# ========= Gradio App =========
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Agentic RAG Chatbot with Your Links")

    with gr.Row():
        links_box = gr.Textbox(
            label="🔗 Enter your links ",
            placeholder="https://example1.com, https://example2.com",
        )

    with gr.Tab("💬 Chat"):
        output_md = gr.Markdown()
        msg = gr.Textbox(label="Put your question here", placeholder="Ask a question...")
        send = gr.Button("Send")

    with gr.Tab("📜 Logs"):
        logs_box = gr.Markdown()

    state_logs = gr.State("")

    send.click(
        chat_fn,
        inputs=[msg, links_box, state_logs],
        outputs=[output_md, msg, logs_box],
    )

demo.launch(share=True)
