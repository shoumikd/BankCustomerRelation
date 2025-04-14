# BankCustomerRelation
This project leverages Retrieval-Augmented Generation (RAG) to create a conversational AI system capable of intelligently answering on topic related to role of Customer Relation Manager. It combines the strengths of large language models with a domain-specific knowledge learny from documents provided.

🚀 Features:
✅ Accurate and contextual responses to banking-related customer queries
🔎 Combines Retrieval with Generation (RAG architecture)
📚 Custom knowledge base with banking-specific documents and FAQs
💬 Conversational agent powered by an LLM
🛠️ Scalable backend for enterprise integration

Architecture Overview:
GenAI
    ├── data/                   # Knowledge base (documents, FAQs)
    ├── agent/ 
        ├── embeddings/             # Vector representations of documents
        ├── retriever/              # FAISS / Elastic / other retriever code
        ├── generator/              # LLM interface (OpenAI, Hugging Face)
    ├── app/                    # Frontend or API layer
    ├── utils/                  # Helper scripts
    ├── README.md
    └── requirements.txt