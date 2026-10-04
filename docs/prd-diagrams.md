# Diagram Mermaid — Investment Research Assistant

Diagram ringkas untuk setiap bagian `prd.md`.

---

## 1. Product Overview

```mermaid
flowchart LR
    A[Analis Investasi] -->|membaca berjam-jam| B[Annual Report<br/>10-K, 10-Q<br/>Earnings Call<br/>News]
    B -->|sulit cari info spesifik| C{Manual Research}
    C -->|dipercepat| D[Investment Research<br/>Assistant]
    D -->|bahasa alami| E[Pertanyaan]
    E -->|jawaban + sitasi| F[Keputusan Investasi<br/>Lebih Cepat]
```

---

## 2. Goal & Success Metrics

```mermaid
flowchart TB
    G[Goal: Jam → Menit] --> M[Success Metrics]
    M --> M1[70% lebih cepat]
    M --> M2[Citation accuracy >95%]
    M --> M3[Response <5 detik]
    M --> M4[Hallucination <2%]
```

---

## 3. Target Users

```mermaid
flowchart TB
    U[Target Users] --> P[Primary]
    U --> S[Secondary]
    P --> P1[Equity Research Analyst]
    P --> P2[Investment Analyst]
    P --> P3[Portfolio Manager]
    S --> S1[Corporate Finance]
    S --> S2[VC Analyst]
    S --> S3[Hedge Fund Analyst]
```

---

## 4. Data Sources

```mermaid
flowchart TB
    DS[Data Sources] --> C[Company Documents]
    DS --> M[Market Information]
    DS --> I[Internal Documents]
    C --> C1[Annual Report / 10-K]
    C --> C2[Quarterly Report / 10-Q]
    C --> C3[Earnings Call Transcript]
    C --> C4[Investor Presentation]
    C --> C5[Sustainability Report]
    M --> M1[SEC Filings]
    M --> M2[OJK Filings]
    M --> M3[Financial News]
    M --> M4[Bloomberg Export]
    M --> M5[Yahoo Finance]
    I --> I1[Previous Research Reports]
    I --> I2[Investment Thesis]
    I --> I3[Internal Notes]
```

---

## 5.1 Document Ingestion

```mermaid
flowchart LR
    U[Upload PDF/DOCX/HTML/TXT] --> P[Extract Text]
    P --> O{Scanned?}
    O -->|Yes| OCR[OCR]
    O -->|No| CL[Clean Formatting]
    OCR --> CL
    CL --> S[Split into Chunks]
```

---

## 5.2 Embedding Pipeline + Metadata

```mermaid
flowchart TB
    A[Annual Report] --> C[Chunking<br/>~500 tokens]
    C --> E[Embedding Model]
    E --> V[(Vector Database)]

    V -.->. M[Metadata]
    M --> M1[Company]
    M --> M2[Year]
    M --> M3[Quarter]
    M --> M4[Page]
    M --> M5[Document Type]
    M --> M6[Section]
```

---

## 5.3 Vector Database

```mermaid
flowchart TB
    V[(Vector DB<br/>Pinecone / Qdrant<br/>Weaviate / Milvus)]
    V --> I[Index berdasarkan]
    I --> I1[Company]
    I --> I2[Industry]
    I --> I3[Year]
    I --> I4[Section]
```

---

## 5.4 Semantic Search

```mermaid
flowchart LR
    Q[Question] --> E[Embedding]
    E --> S[Similarity Search]
    S --> K[Top K Chunks]
    K --> L[LLM]
```

---

## 5.5 Answer Generation

```mermaid
flowchart LR
    L[LLM] --> O[Output]
    O --> O1[Natural Language Answer]
    O --> O2[Bullet Summary]
    O --> O3[Supporting Evidence]
    O --> O4[Citations]
    O4 --> S[Sources:<br/>Page 42<br/>Timestamp 12:41]
```

---

## 5.6 Multi-document Reasoning

```mermaid
flowchart LR
    Q[Compare NVIDIA AI<br/>Strategy 2024 vs 2025] --> R[Retrieve]
    R --> D1[Annual Report 2024]
    R --> D2[Annual Report 2025]
    R --> D3[Earnings Call Q4]
    R --> D4[Investor Day]
    D1 & D2 & D3 & D4 --> RE[Reasoning]
```

---

## 5.7–5.9 Report & Earnings Chat Features

```mermaid
flowchart TB
    F[Fitur]
    F --> F1[5.7 Summarization<br/>Revenue, EPS, Cash Flow,<br/>Risks, Outlook]
    F --> F2[5.8 Investment Thesis<br/>Overview, Growth, Risks,<br/>Valuation, Catalysts]
    F --> F3[5.9 Earnings Call Chat]
    F3 --> T[Transcript] --> RE[Retrieve] --> A[LLM Answer]
```

---

## 5.10 Source Citation

```mermaid
flowchart LR
    A[Setiap Jawaban] --> Wajib[Wajib ada]
    Wajib --> C1[Document]
    Wajib --> C2[Page]
    Wajib --> C3[Paragraph]
    Wajib --> C4[Link]
```

---

## 6. Non-Functional Requirements

```mermaid
flowchart LR
    N[NFR] --> N1[Latency < 5s]
    N --> N2[Availability 99.9%]
    N --> N3[Max 1000 PDF]
    N --> N4[100 Concurrent Users]
```

---

## 7. RAG Pipeline

```mermaid
flowchart TB
    U[Upload PDF] --> DP[Document Parser]
    DP --> TC[Text Cleaning]
    TC --> CE[Chunking Engine]
    CE --> EM[Embedding Model]
    EM --> VD[(Vector Database)]

    UQ[User Question] --> QE[Embedding]
    QE --> VS[Vector Search]
    VD --> VS
    VS --> TK[Top K Relevant Chunks]
    TK --> CB[Context Builder]
    CB --> LLM[LLM]
    LLM --> ANS[Answer + Citations]
```

---

## 8. Recommended Tech Stack

```mermaid
flowchart TB
    subgraph Frontend
        FE[Next.js + React]
    end
    subgraph Backend
        BE[FastAPI]
        Q[Queue: Celery / Redis]
        DB[(PostgreSQL)]
        ST[Storage: S3 / GCS]
    end
    subgraph AI Layer
        LLM[LLM: GPT-5 / Claude 4]
        EMB[Embeddings: text-embedding-3-large]
        FW[Framework: LlamaIndex / LangChain]
        RR[Reranker]
    end
    subgraph Document Processing
        DP[Parsing: Unstructured.io<br/>PyMuPDF / Docling]
        OCR[OCR: Tesseract / Azure DI]
    end
    subgraph Vector Store
        VDB[(Qdrant / Pinecone)]
    end

    FE <--> BE
    BE --> Q
    BE --> DB
    BE --> ST
    BE --> FW
    FW --> LLM
    FW --> EMB
    FW --> RR
    FW --> VDB
    DP --> OCR
    DP --> VDB
```

---

## 9. Future Enhancements

```mermaid
flowchart TB
    F[Future]
    F --> F1[Multi-company Comparison]
    F --> F2[Portfolio-level Q&A]
    F --> F3[Real-time Market Data]
    F --> F4[AI Watchlist + Notifikasi]
    F --> F5[Multimodal Vision<br/>Grafik, Tabel, Chart]
```

---

## Example User Journey

```mermaid
flowchart LR
    S1[1. Upload<br/>Annual Report +<br/>Earnings +<br/>Presentation] --> S2[2. Extract +<br/>Embed +<br/>Store]
    S2 --> S3[3. Ask:<br/>Main revenue<br/>drivers 2025?]
    S3 --> S4[4. RAG Retrieve<br/>relevant chunks]
    S4 --> S5[5. LLM Answer +<br/>Summary +<br/>Citations]
    S5 --> S6[6. Verifikasi via<br/>source asli]
```

---

## V2 Architecture — Production

```mermaid
flowchart TB
    UQ[User Query] --> CM[Conversation Manager]
    CM --> QU[Query Understanding Agent<br/>Entity + Intent + Time Range]
    QU --> RF{Route}

    RF -->|Structured| SD[Structured Data Route<br/>Postgres / TimescaleDB]
    RF -->|Document| DR[Document Route<br/>Vector Database RAG]

    SD --> FT[Financial Metrics Tool<br/>Revenue, EPS, PE]
    DR --> HR[Hybrid Retrieval<br/>BM25 + Dense + Metadata]
    HR --> PR[Parent Retrieval]

    FT & PR --> RR[Re-ranking<br/>Cross Encoder]
    RR --> CA[Context Assembly<br/>Tables + Text + History]
    CA --> LLM[Financial Reasoning LLM]
    LLM --> CV[Citation Verification Agent]
    CV --> FR[Final Response]
```

---

## V2 — Yang Diubah dari PRD

```mermaid
flowchart TB
    C[Perubahan Utama]
    C --> C1[Dual Knowledge System<br/>Structured DB + Vector DB]
    C --> C2[Financial Table Extraction<br/>Layout-aware → Markdown/JSON]
    C --> C3[Metadata Filtering<br/>Entity + Year + Quarter]
    C --> C4[Hybrid Retrieval<br/>Dense + BM25 + Reranker]
    C --> C5[Parent-Child Chunking<br/>Section → Paragraph → Sentence]
    C --> C6[Multi-Agent<br/>Retriever → Analyst →<br/>Citation → Formatter]
    C --> C7[Citation Verification<br/>NLI check]
    C --> C8[Tool Calling<br/>LLM call SQL tool<br/>untuk angka]
    C --> C9[Conversation Memory]
    C --> C10[Evaluation Pipeline<br/>Faithfulness, Precision,<br/>Recall, Numeric Accuracy]
```

---

## V2 — Implementation Phases

```mermaid
flowchart LR
    P1[Phase 1: MVP] --> P2[Phase 2: Production] --> P3[Phase 3: Enterprise]

    P1 --> P1a[Layout-aware parsing]
    P1 --> P1b[Hybrid search BM25+Dense]
    P1 --> P1c[Metadata filtering]
    P1 --> P1d[Parent-child chunking]
    P1 --> P1e[Citation support]

    P2 --> P2a[Structured financial DB]
    P2 --> P2b[Tool calling SQL]
    P2 --> P2c[Conversation memory]
    P2 --> P2d[Reranker]
    P2 --> P2e[RAG evaluation framework]
    P2 --> P2f[Document lifecycle]

    P3 --> P3a[Multi-agent orchestration]
    P3 --> P3b[Citation verification]
    P3 --> P3c[Table reasoning]
    P3 --> P3d[Vision LLM charts]
    P3 --> P3e[Real-time market data]
    P3 --> P3f[RBAC + Multi-tenancy + Audit]
```
