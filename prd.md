# PRD — Investment Research Assistant (LLM + RAG Pipeline)

## 1. Product Overview

### Product Name

**Investment Research Assistant**

### Problem Statement

Investment analysts menghabiskan banyak waktu membaca annual report (100–400 halaman), quarterly reports, earnings call transcripts, dan berita sebelum membuat keputusan investasi. Proses ini memakan waktu berjam-jam dan sulit menemukan informasi spesifik secara cepat.

Tujuan produk adalah memungkinkan analis **bertanya menggunakan bahasa alami** dan mendapatkan jawaban yang akurat dengan referensi langsung ke dokumen sumber.

---

# 2. Goal

Mengurangi waktu research perusahaan dari beberapa jam menjadi beberapa menit dengan memanfaatkan **LLM + Retrieval-Augmented Generation (RAG).**

Success Metrics

* 70% lebih cepat mencari informasi
* Citation accuracy >95%
* Response <5 detik
* Hallucination <2%

---

# 3. Target Users

Primary

* Equity Research Analyst
* Investment Analyst
* Portfolio Manager

Secondary

* Corporate Finance
* VC Analyst
* Hedge Fund Analyst

---

# 4. Data Sources

### Company Documents

* Annual Reports (10-K)
* Quarterly Reports (10-Q)
* Earnings Call Transcript
* Investor Presentation
* Sustainability Report

### Market Information

* SEC Filings
* OJK Filings
* Financial News
* Bloomberg Export
* Yahoo Finance Historical Data

### Internal Documents

* Previous Research Reports
* Investment Thesis
* Internal Notes

---

# 5. Functional Requirements

## 5.1 Document Ingestion

User dapat meng-upload:

* PDF
* DOCX
* HTML
* TXT

System akan

* Extract text
* OCR jika scanned PDF
* Clean formatting
* Split menjadi chunks

---

## 5.2 Embedding Pipeline

Setiap chunk akan

```
Annual Report

↓

Chunking
(~500 tokens)

↓

Embedding Model

↓

Vector Database
```

Metadata

```
Company
Year
Quarter
Page
Document Type
Section
```

Contoh

```
Apple

2025

Annual Report

Page 143

Risk Factors
```

---

## 5.3 Vector Database

Contoh

```
Pinecone

Qdrant

Weaviate

Milvus
```

Index berdasarkan

```
Company

Industry

Year

Section
```

---

## 5.4 Semantic Search

User bertanya

> Why did Tesla's gross margin decline in 2025?

Pipeline

```
Question

↓

Embedding

↓

Similarity Search

↓

Top K Chunks

↓

LLM
```

---

## 5.5 Answer Generation

LLM menghasilkan

* Natural language answer
* Bullet summary
* Supporting evidence
* Citations

Output

```
Gross margin declined mainly because:

• Price reductions across vehicle lineup

• Higher battery production costs

• Lower factory utilization

Sources

Annual Report 2025
Page 42

Earnings Call Q4
Timestamp 12:41
```

---

## 5.6 Multi-document Reasoning

Pertanyaan

> Compare NVIDIA's AI strategy in 2024 vs 2025.

System mengambil

* Annual Report 2024
* Annual Report 2025
* Earnings Call Q4
* Investor Day

Kemudian melakukan reasoning.

---

## 5.7 Financial Report Summarization

Contoh

```
Summarize Apple's Q2 earnings.
```

Output

* Revenue
* EPS
* Cash Flow
* Risks
* Outlook

---

## 5.8 Investment Thesis Generator

Prompt

```
Generate an investment thesis for Microsoft.
```

Output

```
Business Overview

Growth Drivers

Risks

Valuation

Competitive Advantage

Catalysts
```

---

## 5.9 Earnings Call Chat

User

```
Did management mention AI demand?
```

System mencari

```
Transcript

↓

Retrieve

↓

LLM Answer
```

---

## 5.10 Source Citation

Setiap jawaban wajib memiliki

* Document
* Page
* Paragraph
* Link

Tidak boleh ada jawaban tanpa source.

---

# 6. Non Functional Requirements

Latency

<5 seconds

Availability

99.9%

Maximum document

1000 PDF

Concurrent users

100

---

# 7. RAG Pipeline

```
                Upload PDF

                     │

                     ▼

             Document Parser

                     │

                     ▼

              Text Cleaning

                     │

                     ▼

               Chunking Engine

                     │

                     ▼

            Embedding Model

                     │

                     ▼

             Vector Database

                     │

                     ▼

────────────────────────────────────

User Question

↓

Embedding

↓

Vector Search

↓

Top K Relevant Chunks

↓

Context Builder

↓

LLM

↓

Answer + Citations
```

---

# 8. Recommended Tech Stack

| Component        | Recommendation                               |
| ---------------- | -------------------------------------------- |
| LLM              | GPT-5 / Claude 4 Sonnet                      |
| Embeddings       | OpenAI text-embedding-3-large                |
| Vector DB        | Qdrant (self-hosted) atau Pinecone (managed) |
| Framework        | LlamaIndex atau LangChain                    |
| Document Parsing | Unstructured.io, PyMuPDF, Docling            |
| OCR              | Tesseract atau Azure Document Intelligence   |
| Backend          | FastAPI                                      |
| Queue            | Celery / Redis                               |
| Storage          | AWS S3 / Google Cloud Storage                |
| Database         | PostgreSQL                                   |
| Frontend         | Next.js + React                              |
| Deployment       | Docker Compose (`infra/docker-compose.yml`), driven by `start.bat` / `stop.bat` |

---

# 9. Future Enhancements

* Multi-company comparison (misalnya Apple vs Microsoft vs NVIDIA).
* Portfolio-level Q&A berdasarkan seluruh perusahaan dalam portofolio.
* Integrasi data pasar real-time (harga saham, earnings estimates, news sentiment).
* AI-generated watchlist dengan notifikasi ketika ada filing, earnings call, atau berita penting.
* Dukungan multimodal untuk membaca grafik, tabel, dan chart langsung dari laporan keuangan menggunakan Vision LLM.

## Example User Journey

1. Analis mengunggah **Annual Report 2025**, **Q1 & Q2 Earnings Call**, dan **Investor Presentation**.
2. Sistem mengekstrak teks, membuat embeddings, dan menyimpannya di vector database.
3. Pengguna bertanya: *"What were the main drivers of revenue growth in 2025?"*
4. RAG mengambil bagian paling relevan dari beberapa dokumen.
5. LLM menyusun jawaban dengan ringkasan, analisis, dan **sitasi halaman/timestamp** dari sumber yang digunakan.
6. Pengguna dapat membuka sumber asli untuk memverifikasi jawaban, sehingga hasil dapat dipercaya dan diaudit.

PRD yang kamu buat sudah memiliki fondasi yang cukup solid untuk sebuah sistem RAG standar. Struktur penulisan data sources, metadata, dan alur pipeline-nya sudah jelas.

Namun, untuk domain **investasi dan finansial**, dokumen yang dihadapi memiliki karakteristik yang sangat unik dan sensitif. Jika pipeline ini diimplementasikan secara *naive* (RAG standar), sistem ini kemungkinan besar akan gagal saat diuji oleh target pengguna seperti *Equity Research Analyst*.

Berikut adalah analisis mendalam mengenai kelemahan PRD tersebut beserta poin-poin critical yang perlu diperbaiki:

---

## 1. Kelemahan Utama (Critical Gaps)

### 🚨 The Financial Table Problem (Penghancuran Data Tabular)

* **Kelemahan:** Di dokumen finansial, insight terbesar ada di lembar *Balance Sheet*, *Income Statement*, dan *Cash Flow*. Strategi chunking standar (~500 tokens) akan memotong tabel di tengah-tengah baris atau kolom. Hasilnya, hubungan antar angka (misal: baris "Revenue" dengan kolom "2025") akan hilang saat diubah menjadi vektor.
* **Solusi untuk PRD:** Ubah strategi *Document Ingestion*. Wajibkan penggunaan **Layout-Aware Parsing** (seperti menggunakan Docling atau Azure Document Intelligence secara optimal) untuk mengekstrak tabel menjadi format terstruktur (Markdown atau JSON HTML) sebelum masuk ke database, atau jadikan kapabilitas *Multimodal/Vision* sebagai core requirement sejak awal, bukan future enhancement.

### 🔍 Naive Retrieval vs. Complex Financial Reasoning

* **Kelemahan:** Skema pencarian di PRD langsung melompat dari *Similarity Search* ke *Top K Chunks* lalu ke LLM. Pendekatan ini hanya bekerja untuk pencarian fakta sederhana (*"Kapan Apple didirikan?"*). Untuk pertanyaan analisis seperti *"Bandingkan strategi NVIDIA 2024 vs 2025"*, top-K similarity search akan mengambil chunk secara acak yang memiliki kemiripan teks tertinggi, sering kali melewatkan konteks finansial yang krusial di halaman lain.
* **Solusi untuk PRD:** Pipa RAG harus ditingkatkan ke **Advanced/Agentic RAG**. Tambahkan arsitektur seperti:
* **Parent-Child Chunking / Hierarchical Retrieval:** Mengambil chunk kecil untuk pencarian vektor, tetapi memberikan konteks paragraf/halaman penuh ke LLM.
* **Hybrid Search:** Gabungkan Vector Search dengan Keyword Search (BM25) karena istilah finansial sering kali sangat spesifik (misal: kode akun atau metrik seperti "EBITDA").
* **LLM Routing / Query Decomposition:** Pertanyaan kompleks dipecah menjadi beberapa sub-query sebelum melakukan retrieval.



### 📐 Definisi Metrik Sukses yang Mengambang

* **Kelemahan:** Kamu menetapkan metrik seperti `Hallucination <2%` dan `Citation accuracy >95%`. Di industri finansial, salah satu angka saja (misal tertukar antara 2.1% dan 21%) bisa berakibat fatal pada keputusan investasi. Selain itu, tidak disebutkan *bagaimana* cara mengukur angka kurang dari 2% tersebut secara objektif sebelum naik ke production.
* **Solusi untuk PRD:** Tambahkan requirement untuk **RAG Evaluation Framework** (seperti Ragas, TruLens, atau LLM-as-a-judge). Tentukan metrik evaluasi yang jelas seperti *Faithfulness* (apakah jawaban benar-benar hanya dari konteks), *Answer Relevance*, dan *Context Precision*.

### 🌐 Konteks Lokal dan Regulasi (OJK Filings)

* **Kelemahan:** Di bagian data sources, kamu memasukkan *OJK Filings*. Laporan keuangan emiten Indonesia di OJK sering kali menggunakan format dwibahasa (ID/EN) dengan istilah akuntansi lokal yang khas. Embedding model standar (`text-embedding-3-large`) terkadang kurang sensitif terhadap singkatan finansial lokal atau istilah spesifik regulasi di Indonesia jika tidak ditangani dengan baik.
* **Solusi untuk PRD:** Tambahkan kebutuhan *Cross-Lingual Alignment* atau pembuatan kamus istilah (*Glossary Mapping*) di tahap Text Cleaning agar sistem tidak bingung memetakan "Laba Bersih" dengan "Net Income".

---

## 2. Kontradiksi & Ambiguitas pada Non-Functional Requirements

| Poin PRD | Masalah / Kontradiksi | Perbaikan yang Disarankan |
| --- | --- | --- |
| **Latency < 5 detik** | Untuk fitur *Multi-document reasoning* dan *Investment Thesis Generator*, LLM harus membaca banyak dokumen sekaligus dan menghasilkan output teks yang panjang. Batas waktu kurang dari 5 detik sangat tidak realistis jika dilakukan secara *synchronous* tanpa *streaming*. | Pisahkan SLA latency: Chat standar (kurang dari 3 detik dengan *streaming*), sedangkan untuk *Thesis Generator* buat menggunakan sistem *asynchronous job* menggunakan Celery dengan notifikasi selesai. |
| **Maximum document 1000 PDF** | Jika ada 100 *concurrent users* yang masing-masing mengelola 15-20 perusahaan (di mana satu perusahaan memiliki puluhan dokumen dari annual report hingga berita), kapasitas total 1000 PDF akan langsung penuh dalam waktu singkat. | Perjelas batasannya: Apakah 1000 PDF itu kuota *per user workspace*, *per company*, atau total kapasitas sistem? Disarankan minimal skala tingkat *workspace* (misal: 500 PDF per workspace). |

---

## 3. Rekomendasi Tambahan pada Tech Stack

* **Metadata Filtering:** Di bagian Vector DB, pastikan memilih database yang mendukung *Pre-filtering* metadata secara cepat (Qdrant dan Pinecone sudah mendukung ini dengan baik). Ini penting agar ketika user bertanya tentang "Apple 2025", sistem langsung mengunci pencarian hanya pada *chunk* yang memiliki metadata `Company: Apple` dan `Year: 2025`, bukan mencari di seluruh ruang vektor global.
* **Chunking Strategy:** Selain berdasarkan token (~500 tokens), tambahkan pendekatan **Semantic Chunking** atau **Document Structure Chunking** (berdasarkan *Header* / *Section* laporan keuangan) agar batasan pembahasan tidak terpotong di tengah jalan.
Berdasarkan analisis terhadap PRD di atas, berikut kelemahan-kelemahan utama yang perlu diperbaiki agar produk dapat diandalkan dan sesuai dengan kebutuhan analis investasi.

### 1. **Ketiadaan Ekstraksi Data Terstruktur (Tabel & Angka)**
PRD saat ini hanya mendeskripsikan ekstraksi teks mentah dari PDF, DOCX, dll. Laporan keuangan sarat dengan **tabel, angka, dan chart** (laporan laba rugi, neraca, arus kas). Pipeline text-only akan:
- Mengabaikan informasi kritis yang tersimpan dalam bentuk tabel.
- Memotong tabel saat chunking, sehingga data numerik menjadi tidak koheren.
- Tidak bisa menjawab pertanyaan kuantitatif seperti *“Berapa EPS Q3 2025?”* secara andal.

👉 **Perbaikan:** Tambahkan *table extraction engine* (misal Camelot, Tabula, atau Unstructured dengan mode ‘hi_res’) yang menyimpan data numerik ke database terstruktur, lalu gabungkan dengan jawaban RAG.

### 2. **Metrik ‘Hallucination <2%’ Tidak Realistis & Sulit Diukur**
Mengklaim halusinasi <2% tanpa definisi operasional dan metode pengukuran adalah jebakan. Model bahasa generatif secara inheren bisa berhalusinasi, terutama jika menggabungkan banyak sumber atau menghasilkan ringkasan keuangan. Tanpa *human-in-the-loop* atau verifikasi fakta otomatis yang kuat, metrik ini hampir mustahil dicapai.

👉 **Perbaikan:** Definisikan hallucination secara spesifik (misal angka salah, klaim tidak berdasar sumber). Bangun dataset *gold standard* dengan jawaban benar dan lakukan evaluasi berkala. Tambahkan modul *fact-checking* atau fallback “Saya tidak tahu” jika keyakinan rendah.

### 3. **Pipeline Retrieval Terlalu Naif, Tidak Ada Hybrid Search & Re-ranking**
Diagram menunjukkan pencarian vektor semata (*dense retrieval*). Ini lemah untuk:
- Pertanyaan eksak (misal “Siapa CFO?”) yang lebih cocok dengan keyword search.
- Kueri spesifik tanggal atau angka.
Tidak disebutkan *re-ranking* untuk meningkatkan presisi chunk yang diteruskan ke LLM.

👉 **Perbaikan:** Adopsi **hybrid search** (vektor + BM25) dan *re-ranker model* (misal Cohere Rerank, cross-encoder) agar konteks yang diberikan ke LLM lebih relevan.

### 4. **Tidak Ada Integrasi Data Terstruktur (Bloomberg, Yahoo Finance)**
Sumber data seperti Bloomberg Export dan Yahoo Finance History berbentuk data tabular/time-series, bukan teks naratif. Memasukkannya ke pipeline chunking teks adalah kekeliruan. Pertanyaan seperti *“Bagaimana pergerakan harga saham setelah earnings call?”* tidak bisa dijawab hanya dari teks.

👉 **Perbaikan:** Pisahkan jalur: data teks masuk RAG, data numerik/time-series disimpan di database khusus (misal PostgreSQL + TimescaleDB) dan diakses melalui tool/function calling oleh LLM.

### 5. **Ketiadaan Penanganan Konteks Percakapan Multi-turn**
Contoh *“Did management mention AI demand?”* adalah pertanyaan lanjutan. Tanpa riwayat percakapan, sistem tidak akan tahu perusahaan mana atau periode laporan yang dimaksud.

👉 **Perbaikan:** Tambahkan manajemen sesi percakapan di backend, serta *query rewriting* yang memperkaya pertanyaan dengan konteks dari history.

### 6. **Metadata Filtering Tidak Diterapkan Saat Retrieval**
Meskipun vector DB diindeks berdasarkan *Company, Year, Section*, pipeline di bagian 5.4 hanya menjalankan *similarity search* mentah. Akibatnya, pertanyaan tentang *Apple* bisa mengambil chunk dari *Microsoft* jika embedding-nya mirip.

👉 **Perbaikan:** Wajibkan pre-filtering metadata (perusahaan, tahun, tipe dokumen) sebelum pencarian vektor, berdasarkan entitas yang diekstrak dari pertanyaan pengguna.

### 7. **Ketiadaan Manajemen Versi & Pembaruan Dokumen**
Analis akan mengunggah laporan baru setiap kuartal. Dokumen lama mungkin perlu ditandai sebagai usang atau dihapus agar tidak mempengaruhi jawaban (*“kenaikan pendapatan tahun 2025”* tidak boleh tercampur data 2024 yang sudah kadaluwarsa). PRD tidak menyinggung siklus hidup dokumen.

👉 **Perbaikan:** Terapkan *document lifecycle management*: kemampuan menghapus atau menonaktifkan embeddings berdasarkan ID dokumen, serta filter default ke dokumen terbaru kecuali diminta spesifik.

### 8. **Ketiadaan Kontrol Akses & Keamanan Data**
Dokumen internal seperti *Previous Research Reports, Investment Thesis* bersifat sangat rahasia. PRD tidak menyebutkan mekanisme otentikasi pengguna, isolasi data antar firma/analis, atau enkripsi.

👉 **Perbaikan:** Desain sistem multi-tenant dengan RBAC (Role-Based Access Control). Semua dokumen dienkripsi saat disimpan, dan akses API dibatasi per user/group.

### 9. **Sitasi Hanya Berupa Anotasi, Bukan Jaminan Kebenaran**
Persyaratan *“setiap jawaban wajib memiliki sumber”* bersifat format, bukan verifikasi kebenaran bahwa sumber yang dikutip benar-benar mendukung klaim. LLM bisa mengarang halaman atau mengutip paragraf yang tidak relevan namun tetap mencantumkan nomor halaman.

👉 **Perbaikan:** Tambahkan *citation verification module* yang membandingkan klaim dalam jawaban dengan isi chunk yang dirujuk menggunakan NLI (Natural Language Inference). Jika tidak terdukung, beri peringatan atau tolak jawaban.

### 10. **Tidak Ada Rancangan Evaluasi & Umpan Balik Berkelanjutan**
Metrik sukses seperti *70% lebih cepat* atau *citation accuracy >95%* tidak akan tercapai tanpa mekanisme pengukuran dan perbaikan iteratif. PRD tidak menyediakan loop feedback dari pengguna (thumbs up/down, koreksi).

👉 **Perbaikan:** Bangun antarmuka rating jawaban, simpan log interaksi, dan lakukan evaluasi offline mingguan untuk mengukur akurasi dan kecepatan. Gunakan feedback untuk fine-tuning prompt atau model retrieval.

Dengan memperbaiki kelemahan-kelemahan di atas, produk akan lebih siap menghadapi realitas riset investasi yang sangat menuntut akurasi, kecepatan, dan kepercayaan.

Saya setuju dengan hampir seluruh analisis tersebut. Itu adalah kritik yang memang sering muncul ketika sebuah **RAG PoC** ingin dinaikkan menjadi **production-grade system** untuk finance.

Kalau saya menyusun ulang PRD untuk level perusahaan seperti Bloomberg, BlackRock, Morningstar, atau bank investasi, saya tidak lagi menyebutnya sebagai **LLM + RAG**, tetapi sebagai **Agentic Financial Intelligence Platform**.

Berikut arsitektur yang menurut saya lebih tepat.

---

# Investment Research Assistant v2 (Production Architecture)

```
                           User Query
                                │
                                ▼
                     Conversation Manager
                                │
                                ▼
                   Query Understanding Agent
        (Entity Extraction + Intent + Time Range)
                                │
          ┌─────────────────────┴────────────────────┐
          │                                          │
          ▼                                          ▼
 Structured Data Route                     Document Route
(Postgres/TimescaleDB)                (Vector Database RAG)
          │                                          │
 Financial Metrics Tool                 Hybrid Retrieval
 (Revenue, EPS, PE, etc.)        (BM25 + Dense + Metadata)
          │                                          │
          │                                   Parent Retrieval
          │                                          │
          ▼                                          ▼
                 Re-ranking (Cross Encoder)
                                │
                                ▼
                  Context Assembly Engine
      (Tables + Text + Previous Conversation)
                                │
                                ▼
                      Financial Reasoning LLM
                                │
                                ▼
                   Citation Verification Agent
                                │
                                ▼
                         Final Response
```

---

# Yang saya ubah dari PRD sebelumnya

## 1. Dual Knowledge System

Ini menurut saya perubahan terbesar.

Jangan semua data dimasukkan ke Vector DB.

Pisahkan menjadi dua.

### A. Structured Database

Berisi angka.

Misalnya

```
Revenue

Net Income

EPS

Operating Margin

Cash Flow

PE Ratio

Debt

Dividend
```

Disimpan di

```
Postgres

DuckDB

TimescaleDB
```

Karena angka tidak cocok dicari menggunakan embedding.

---

### B. Vector Database

Berisi

```
Management Discussion

Risk Factors

Business Strategy

Annual Report

Transcript

News

Research Notes

Conference Calls
```

Karena ini memang berbentuk teks.

---

## 2. Financial Table Extraction

Ini menurut saya wajib.

Pipeline ingestion menjadi

```
PDF

↓

Layout Detection

↓

Table Detection

↓

Table Extraction

↓

Markdown Table

↓

Embedding
```

atau

```
JSON
```

bukan text biasa.

Contoh

Daripada

```
Revenue
2025
120
```

menjadi

```
Revenue | 2025 | 2024

120     | 110
```

Hubungan antar kolom tetap terjaga.

---

## 3. Metadata Filtering

PRD sebelumnya langsung

```
Question

↓

Embedding

↓

Similarity Search
```

Ini terlalu sederhana.

Harusnya

```
Question

↓

Extract Company

↓

Extract Year

↓

Extract Quarter

↓

Metadata Filter

↓

Hybrid Search
```

Misalnya

```
Compare Apple and Microsoft Q2 2025
```

langsung menjadi

```
Company

Apple

Microsoft

Year

2025

Quarter

Q2
```

baru retrieval.

---

## 4. Hybrid Retrieval

Jangan hanya embedding.

```
Dense Retrieval

+

BM25

+

Metadata Filter
```

baru

```
Reranker
```

Misalnya

```
EBITDA

GAAP

ASC606

IFRS15
```

lebih mudah ditemukan lewat keyword daripada embedding.

---

## 5. Parent-Child Chunking

Bukan

```
500 tokens
```

tetapi

```
Annual Report

↓

Section

↓

Paragraph

↓

Sentence
```

Vector search dilakukan pada chunk kecil.

Tetapi LLM diberi

```
1 halaman penuh

atau

1 section penuh
```

supaya reasoning lebih baik.

---

## 6. Multi-Agent

Daripada satu LLM mengerjakan semuanya.

Pisahkan.

```
Retriever Agent

↓

Financial Analyst Agent

↓

Citation Checker

↓

Response Formatter
```

Lebih stabil.

---

## 7. Citation Verification

Ini sangat penting.

Jawaban

```
Revenue increased because...

```

harus dicek lagi

```
Apakah benar chunk ini mengatakan begitu?
```

Menggunakan

```
NLI

atau

Small LLM
```

Kalau tidak didukung source

langsung

```
Reject Answer
```

---

## 8. Tool Calling

Daripada LLM menghafal angka.

Misalnya user bertanya

> What's Apple's EPS in Q2 2025?

LLM cukup memanggil

```
SQL Tool
```

```
SELECT eps

FROM earnings

WHERE

company='Apple'

AND quarter='Q2'

AND year=2025
```

baru menjawab.

Ini jauh lebih akurat.

---

## 9. Conversation Memory

Harus ada.

User

```
Tell me about Apple.
```

↓

```
How about Microsoft?
```

↓

```
Compare them.
```

↓

```
Which company has better margins?
```

Kalau tidak ada memory,

RAG akan gagal.

---

## 10. Evaluation Pipeline

Saya juga akan menambahkan satu komponen khusus.

```
Offline Evaluation
```

setiap deployment.

Misalnya

```
500 benchmark questions

↓

Run RAG

↓

Evaluate
```

menggunakan

* Faithfulness
* Context Precision
* Context Recall
* Answer Relevance
* Citation Precision
* Numeric Accuracy (khusus finance)

Kalau skornya turun

deployment diblok.

---

# Prioritas implementasi

Kalau saya yang menjadi Tech Lead, saya akan membagi roadmap menjadi tiga fase.

### Phase 1 (MVP)

* Layout-aware document parsing
* Hybrid search (BM25 + Dense)
* Metadata filtering
* Parent-child chunking
* Citation support

### Phase 2 (Production)

* Structured financial database
* Tool calling (SQL)
* Conversation memory
* Reranker
* RAG evaluation framework
* Document lifecycle management

### Phase 3 (Enterprise)

* Multi-agent orchestration
* Citation verification
* Table reasoning
* Chart/image understanding (Vision LLM)
* Real-time market data integration
* Role-based access control (RBAC), multi-tenancy, encryption, and audit logging

Dengan arsitektur seperti ini, sistem tidak lagi hanya menjadi chatbot berbasis dokumen, tetapi benar-benar berfungsi sebagai **AI Investment Research Analyst** yang mampu menggabungkan reasoning atas dokumen, data numerik, konteks percakapan, dan sumber yang dapat diaudit—sesuai kebutuhan institusi keuangan yang menuntut akurasi tinggi.
