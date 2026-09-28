# Comprehensive Guide & Reference Notes: Enterprise RAG & Agentic Systems

**Date:** September 28, 2026  
**Topics Covered:** Core RAG Concepts, Document Handling, Stateful Tooling, Similarity Mechanics, LangGraph Orchestration, and System Evaluation.

---

## 1. Embeddings & Vector Stores (The Core RAG Engine)

In a traditional database, search relies on exact keyword matching (for example, searching for `"salary"` will fail to match `"compensation"`). Retrieval-Augmented Generation (RAG) resolves this limitation using **semantic search**:

* **Text Embeddings (`GoogleGenerativeAIEmbeddings`):** A mathematical transformation that converts unstructured text into a dense numerical vector (an array of floating-point numbers). Sentences, phrases, or documents with similar semantic meanings end up close to one another in this high-dimensional vector space.
  * *Implementation:* `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
* **Vector Store (`FAISS`):** **F**acebook **A**I **S**imilarity **S**earch is an in-memory vector database designed to index dense vectors and perform high-speed distance searches (such as Cosine or Euclidean distance) to match query vectors with document vectors.
  * *Implementation:* `FAISS.from_documents(...)` or `FAISS.from_texts(...)`

---

## 2. Document Abstraction (`langchain_core.documents.Document`)

LangChain standardizes all text chunks indexed inside or retrieved from vector stores using the unified `Document` object:

* **Page Content (`page_content`):** The raw string segment extracted from the source text or document file.
* **Metadata (`metadata`):** A Python dictionary containing contextual attributes (e.g., `{"source": "HR_Policy.pdf", "page": 4}`). This allows tools and agents to cite sources accurately in their final responses.

---

## 3. Tool Construction & Function Calling (`@tool`)

Rather than injecting full document sets directly into the system prompt, the agent is provided with **Tools**:

* **The `@tool` Decorator:** Converts a standard Python function into a schema-defined capability that LLMs can inspect and invoke.
* **Docstring Significance:** The underlying model determines *when* and *why* to invoke a tool based on its docstring description. A clear docstring (e.g., `"""Search private internal company documents..."""`) signals to the model to invoke that specific tool when relevant queries arise.

---

## 4. Dynamic Tool State Management (Live Knowledge Base Updates)

Unlike static tool configurations that only perform read operations, **stateful tools** interact with running runtime objects:

* Tools such as `add_document_to_knowledge_base` interact directly with the active in-memory `vector_store` instance.
* When a user provides new context or requests the system to store information, the model calls this tool to embed the text on the fly and push it into the active FAISS index—making it searchable immediately within the same execution session.

---

## 5. Vector Similarity Search Mechanics

When a retrieval tool executes `vector_store.similarity_search(query, k=3)`, the following operational sequence occurs:

1. The raw text query string is passed into the tool by the model node.
2. The query string is converted into a vector via the embedding model (`models/embedding-001`).
3. FAISS computes mathematical distance metrics comparing the query vector against stored document vectors.
4. The parameter `k=3` instructs FAISS to return the top 3 nearest `Document` objects.
5. The tool formats the returned text chunks and metadata, passing them back to the model inside a `ToolMessage`.

---

## 6. LangGraph Execution & Routing Mechanics

The execution lifecycle operates as a stateful, event-driven loop:

```text
    ┌──────────────┐
    │  User Query  │
    └──────┬───────┘
           │
           ▼
┌──────────────────────┐      Calls Tool      ┌──────────────────────┐
│                      ├─────────────────────►│     Tools Node       │
│      LLM Node        │                      │ (e.g., FAISS Search) │
│ (Agent Reasoning)    │◄─────────────────────┤                      │
└──────────┬───────────┘    Returns Context   └──────────────────────┘
           │
           │ Final Answer Generated
           ▼
    ┌──────────────┐
    │    Output    │
    └──────────────┘
