# Hallucination Detection Pipeline — Full Architecture Diagram

This document contains comprehensive Mermaid diagrams covering every layer of the project:
- **Layer 1:** The 13-step core pipeline (data flow)
- **Layer 2:** Models layer (HuggingFace wrappers)
- **Layer 3:** Retrieval layer (FAISS + BM25)
- **Layer 4:** API layer (FastAPI + Flask)
- **Layer 5:** Frontend pages and JS modules
- **Layer 6:** Scripts / CLI entry points
- **Layer 7:** Evaluation & data flow
- **Layer 8:** Full end-to-end system view

All diagrams use the official [Mermaid](https://mermaid.js.org/) syntax and can be rendered by GitHub, VS Code, Obsidian, or any Mermaid-compatible viewer.

---

## Layer 1 — The 13-Step Core Pipeline

This is the heart of the project. Every step is one Python module under `src/steps/`.

```mermaid
flowchart TD
    Start([User Query]) --> S1

    %% ====== STEP 1: QUERY UNDERSTANDING ======
    subgraph S1["Step 1 — Query Understanding"]
        S1A[Intent Classification<br/>heuristic] --> S1B[spaCy NER<br/>en_core_web_sm]
        S1A --> S1C[GLiNER NER<br/>knowledgator/gliner-bi-base-v1.0]
        S1A --> S1D[LLM Query Rewrite<br/>Llama 3.1 / Qwen]
        S1B & S1C & S1D --> S1E[Entities + Variants]
    end
    S1E --> S2

    %% ====== STEP 2: GROUNDED GENERATION ======
    subgraph S2["Step 2 — Grounded Generation"]
        S2A[Backend Selector<br/>llama / qwen] --> S2B[LlamaGenerator<br/>meta-llama/Meta-Llama-3.1-8B-Instruct]
        S2A --> S2C[QwenGenerator<br/>Qwen/Qwen2.5-1.5B-Instruct]
        S2B & S2C --> S2D[Prompt:<br/>Context + Question]
        S2D --> S2E[Initial Answer]
    end
    S2E --> S3

    %% ====== STEP 3: EARLY DETECTION ======
    subgraph S3["Step 3 — Early Hallucination Detection"]
        S3A[BGE-M3 Embedder<br/>BAAI/bge-m3] --> S3B[Cosine Similarity<br/>q × answer]
        S3A --> S3C[Feature Vector:<br/>cosine, claim_count,<br/>overlap, answer_length]
        S3C --> S3D[XGBoost Classifier<br/>or sklearn-logreg]
        S3D --> S3E[Hallucination Probability]
    end
    S3E --> S4

    %% ====== STEP 4: CONFIDENCE ASSESSMENT ======
    subgraph S4["Step 4 — Confidence Assessment (EEDC)"]
        S4A[NLI Entropy H] --> S4B[EEDC Platt Scorer<br/>α·H + β·(1-top1) + γ·(1-cons) + δ]
        S4B --> S4C[Temperature Scaler<br/>grid search T ∈ [0.25, 4.0]]
        S4C --> S4D[Calibrated Score]
    end
    S4D --> S5

    %% ====== STEP 5: ATOMIC CLAIM EXTRACTION ======
    subgraph S5["Step 5 — Atomic Claim Extraction"]
        S5A[Rule-based Splitter<br/>regex sentence] --> S5E[Claims + Provenance]
        S5A --> S5B{Mode?}
        S5B -->|llm| S5C[LLM Splitter<br/>Llama 3.1]
        S5B -->|hybrid| S5D[Rule → LLM fallback]
        S5C & S5D --> S5E
    end
    S5E --> S6

    %% ====== STEP 6: HYBRID RETRIEVAL ======
    subgraph S6["Step 6 — Hybrid Retrieval"]
        S6A[Query Embedding<br/>BGE-M3] --> S6B[FAISS Index<br/>IndexFlatIP]
        S6A --> S6C[BM25 Index<br/>pure-python]
        S6B --> S6D[RRF Fusion<br/>k=60]
        S6C --> S6D
        S6D --> S6E[Top-K Evidence]
    end
    S6E --> S7

    %% ====== STEP 7: RERANKING ======
    subgraph S7["Step 7 — Cross-Encoder Reranking"]
        S7A[BGE-Reranker-v2<br/>BAAI/bge-reranker-v2-m3] --> S7B[Relevance Scores]
        S7B --> S7C[Sort + Top-K]
    end
    S7C --> S8

    %% ====== STEP 8: NLI VERIFICATION ======
    subgraph S8["Step 8 — NLI Verification"]
        S8A[DeBERTa-v3-MNLI<br/>mDeBERTa-xnli-multilingual] --> S8B[For each claim × evidence]
        S8B --> S8C{Label?}
        S8C -->|entail| S8D[supported]
        S8C -->|contradict| S8E[contradicted]
        S8C -->|neutral| S8F[unknown]
        S8D & S8E & S8F --> S8G[Per-claim Verdicts]
    end
    S8G --> S9

    %% ====== STEP 9: CONFIDENCE FUSION ======
    subgraph S9["Step 9 — Confidence Fusion"]
        S9A[Weights:<br/>sim 0.30 / ret 0.20<br/>nli 0.25 / cov 0.15<br/>eedc 0.10] --> S9B[Weighted Sum]
        S9B --> S9C[Per-claim Score]
        S9B --> S9D[Overall Confidence]
    end
    S9D --> S10

    %% ====== STEP 10: REGENERATION ======
    subgraph S10["Step 10 — Evidence-Guided Regeneration"]
        S10A{Verdicts} -->|bad claims| S10B{Mode?}
        S10B -->|stub| S10C[KEEP]
        S10B -->|evidence| S10D[Join supported claims]
        S10B -->|regenerate| S10E[Llama 3.1 rewrite<br/>+ verified evidence]
        S10D & S10E --> S10F[Final Answer Candidate]
    end
    S10F --> S11

    %% ====== STEP 11: FINAL GATE ======
    subgraph S11["Step 11 — Final Verification Gate"]
        S11A[Confidence score] --> S11B{conf ≥ 0.70<br/>AND<br/>unsupp == 0?}
        S11B -->|yes| S11C[✅ PASS]
        S11B -->|no, conf < 0.30| S11D[❌ ABSTAIN]
        S11B -->|gray zone| S11E{attempts &lt; max?}
        S11E -->|yes| S11F[🔄 REGEN]
        S11E -->|no| S11G[best-of PASS/ABSTAIN]
    end
    S11C & S11D & S11F & S11G --> S12

    %% ====== STEP 12: EXPLAINABLE RESPONSE ======
    subgraph S12["Step 12 — Explainable Response"]
        S12A[JSON Payload<br/>query, answer, claims,<br/>evidence, confidence, decision] --> S12C[Output]
        S12B[Markdown Report<br/>per-claim verdicts +<br/>evidence + decision log] --> S12C
    end
    S12C --> S13

    %% ====== STEP 13: EVALUATION ======
    subgraph S13["Step 13 — Evaluation"]
        S13A[Generation Metrics:<br/>BLEU, ROUGE-L, BERTScore] --> S13F[Aggregate Report]
        S13B[Retrieval Metrics:<br/>R@K, P@K, MRR, nDCG] --> S13F
        S13C[Detection Metrics:<br/>Acc, P/R/F1, ROC-AUC] --> S13F
        S13D[Claim Metrics:<br/>Precision, Recall, F1] --> S13F
        S13E[Pipeline Metrics:<br/>HRR, CCR, VSR] --> S13F
    end
    S13F --> End([PipelineResult +<br/>data/metrics.json])

    %% ====== STYLING ======
    classDef step fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#fff
    classDef sub fill:#0f766e,stroke:#5eead4,stroke-width:1.5px,color:#fff
    classDef decision fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fff
    classDef terminal fill:#166534,stroke:#86efac,stroke-width:2px,color:#fff
    classDef model fill:#4c1d95,stroke:#c4b5fd,stroke-width:1.5px,color:#fff
    class S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13 step
    class S1A,S1B,S1C,S1D,S1E,S2A,S2B,S2C,S2D,S2E,S3A,S3B,S3C,S3D,S3E,S4A,S4B,S4C,S4D,S5A,S5B,S5C,S5D,S5E,S6A,S6B,S6C,S6D,S6E,S7A,S7B,S7C,S8A,S8B,S8C,S8D,S8E,S8F,S8G,S9A,S9B,S9C,S9D,S10A,S10B,S10C,S10D,S10E,S10F,S12A,S12B,S12C,S13A,S13B,S13C,S13D,S13E,S13F sub
    class S11B,S11E decision
    class Start,End terminal
    class S2B,S2C,S3A,S5C,S6B,S7A,S8A model
```

---

## Layer 2 — Models Layer (HuggingFace Wrappers)

```mermaid
flowchart LR
    subgraph PIPELINE[Pipeline / src/pipeline.py]
        P[Pipeline Orchestrator]
    end

    subgraph MODELS[src/models/ — 12 HF Wrappers]
        direction TB
        M_EMB[BGE_M3Embedder<br/>BAAI/bge-m3]
        M_LLM[LlamaGenerator<br/>meta-llama/Meta-Llama-3.1-8B-Instruct]
        M_QWEN[QwenGenerator<br/>Qwen/Qwen2.5-1.5B-Instruct]
        M_OR[OpenRouterGenerator<br/>openai/gpt-oss-20b:free]
        M_RR[BGEReranker<br/>BAAI/bge-reranker-v2-m3]
        M_NLI[DebertaNLIModel<br/>MoritzLaurer/mDeBERTa-xnli-...]
        M_GLINER[GLiNERWrapper<br/>knowledgator/gliner-bi-base-v1.0]
        M_SPACY[SpacyNER<br/>en_core_web_sm]
        M_XGB[XGBClassifier<br/>xgboost / sklearn-logreg]
        M_EEDC[EEDC<br/>Platt sigmoid]
        M_TS[TemperatureScaler<br/>grid search T]
        M_ISO[IsotonicCalibrator<br/>pool-adjacent-violators]
        M_REG[registry.py<br/>build_status_report]
    end

    P --> M_EMB
    P --> M_LLM
    P --> M_QWEN
    P --> M_OR
    P --> M_RR
    P --> M_NLI
    P --> M_GLINER
    P --> M_SPACY
    P --> M_XGB
    P --> M_EEDC
    P --> M_TS
    P --> M_ISO
    M_REG -.probes.-> M_LLM
    M_REG -.probes.-> M_EMB
    M_REG -.probes.-> M_RR
    M_REG -.probes.-> M_NLI
    M_REG -.probes.-> M_GLINER
    M_REG -.probes.-> M_TS

    classDef modelBox fill:#312e81,stroke:#a78bfa,stroke-width:2px,color:#fff
    classDef regBox fill:#7f1d1d,stroke:#fca5a5,stroke-width:2px,color:#fff
    class M_EMB,M_LLM,M_QWEN,M_OR,M_RR,M_NLI,M_GLINER,M_SPACY,M_XGB,M_EEDC,M_TS,M_ISO modelBox
    class M_REG regBox
```

---

## Layer 3 — Retrieval Layer (FAISS + BM25 Hybrid)

```mermaid
flowchart TD
    Q[Query String] --> EMB[BGE-M3 Encode<br/>→ 1024-d vector]
    EMB --> QV[Query Vector]

    subgraph FAISS[FAISSIndex]
        F1[Add: doc_ids + docs + vecs]
        F2[Normalize L2]
        F3[faiss.IndexFlatIP]
        F4[Top-K Dense Hits]
    end

    subgraph BM25[BM25Index — pure Python]
        B1[Add: doc_ids + docs]
        B2[Tokenize + DF + avgdl]
        B3[Robertson IDF]
        B4[BM25 TF formula]
        B5[Top-K Sparse Hits]
    end

    subgraph RRF[HybridRetriever]
        R1[Reciprocal Rank Fusion<br/>1 / k + rank + 1]
        R2[Sort by RRF score]
        R3[Final Top-K Evidence]
    end

    QV --> F1
    F1 --> F2 --> F3 --> F4
    Q --> B1
    B1 --> B2 --> B3 --> B4 --> B5
    F4 --> R1
    B5 --> R1
    R1 --> R2 --> R3

    R3 --> OUT[Evidence list<br/>doc_id, text, score]

    classDef idx fill:#0e7490,stroke:#67e8f9,color:#fff
    classDef fusion fill:#9d174d,stroke:#f9a8d4,color:#fff
    class FAISS,BM25 idx
    class RRF fusion
```

---

## Layer 4 — API Layer (FastAPI + Flask)

```mermaid
flowchart TB
    subgraph CLIENTS[Clients]
        C1[curl / PowerShell]
        C2[Web Browser]
        C3[Python / JS SDK]
    end

    subgraph FAST[FastAPI Service — api/app.py — :8000]
        F1[GET /health]
        F2[POST /index]
        F3[POST /query]
        F4[GET /metrics]
        F5[GET /structure]
        F6[GET /models/status]
        F_LIFESPAN[asyncio.to_thread<br/>Pipeline singleton]
    end

    subgraph FLASK[Flask Frontend — frontend/app.py — :5000]
        L1[GET /]
        L2[GET /structure]
        L3[GET /evaluation]
        L4[GET /literature]
        L5[GET /api/health]
        L6[POST /api/index]
        L7[POST /api/query]
        L8[GET /api/metrics]
        L9[GET /api/evaluation]
        L10[GET /api/evaluation/datasets]
        L11[GET /api/evaluation/comparison]
        L12[GET /api/accuracy]
        L13[GET /models/status]
        L14[GET /api/literature/comparison]
        L15[GET /api/literature/why-we-win]
        L16[GET /api/literature/headline]
        L17[GET /api/structure]
        L_LAZY[get_pipeline<br/>Lazy bootstrap]
    end

    C1 --> F1 & F2 & F3 & F4 & F5 & F6
    C2 --> L1 & L2 & L3 & L4
    C3 --> L5 & L6 & L7 & L8 & L9 & L10 & L11 & L12 & L13 & L14 & L15 & L16 & L17

    F1 & F2 & F3 & F4 & F5 & F6 --> F_LIFESPAN
    L5 & L6 & L7 & L8 & L9 & L10 & L11 & L12 & L13 & L14 & L15 & L16 & L17 --> L_LAZY

    F_LIFESPAN --> PIPE[src/pipeline.Pipeline]
    L_LAZY --> PIPE

    classDef fast fill:#1e40af,stroke:#93c5fd,color:#fff
    classDef flask fill:#15803d,stroke:#86efac,color:#fff
    classDef client fill:#525252,stroke:#d4d4d4,color:#fff
    class F1,F2,F3,F4,F5,F6,F_LIFESPAN fast
    class L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,L_LAZY flask
    class C1,C2,C3 client
```

---

## Layer 5 — Frontend Pages & JavaScript Modules

```mermaid
flowchart LR
    subgraph TPL[Templates — frontend/templates/]
        T1[index.html<br/>Chat + Trace + Accuracy + Lit Card]
        T2[structure.html<br/>Project tree]
        T3[evaluation.html<br/>Dataset pills + per-query drill-down]
        T4[literature.html<br/>Verdict matrix + headline]
    end

    subgraph JS[JavaScript — frontend/static/js/]
        J1[app.js — 23 KB<br/>Main chat panel]
        J2[structure.js — 2.8 KB<br/>Tree filter + depth]
        J3[evaluation.js — 17 KB<br/>Per-query + charts]
        J4[literature.js — 9.3 KB<br/>Verdict rendering]
    end

    subgraph CSS[Stylesheets — frontend/static/css/]
        C1[style.css — 17.6 KB]
        C2[literature.css — 4.2 KB]
    end

    T1 --> J1
    T2 --> J2
    T3 --> J3
    T4 --> J4
    T1 & T2 & T3 & T4 --> C1
    T4 --> C2

    J1 --> API1[/api/query, /api/evaluation,<br/>/api/accuracy, /api/literature/*/]
    J2 --> API2[/api/structure/]
    J3 --> API3[/api/evaluation/datasets,<br/>/api/evaluation/comparison/]
    J4 --> API4[/api/literature/comparison,<br/>/api/literature/why-we-win,<br/>/api/literature/headline/]

    classDef tplBox fill:#7c2d12,stroke:#fdba74,color:#fff
    classDef jsBox fill:#0f766e,stroke:#5eead4,color:#fff
    classDef cssBox fill:#581c87,stroke:#d8b4fe,color:#fff
    class T1,T2,T3,T4 tplBox
    class J1,J2,J3,J4 jsBox
    class C1,C2 cssBox
```

---

## Layer 6 — Scripts / CLI Entry Points

```mermaid
flowchart TD
    USER([User / CI]) --> PICK{Goal?}

    PICK -->|Single query| RI[run_inference.py<br/>--query '...']
    PICK -->|Markdown report| RPT[report.py<br/>--query '...']
    PICK -->|Build index| BI[build_index.py<br/>--input data.jsonl<br/>--out data/index]
    PICK -->|Evaluate BASE| EV[evaluate.py<br/>--synth / --dataset / --data]
    PICK -->|Evaluate FULL| EVF[evaluate_full.py<br/>+ detection P/R/F1 + ROC-AUC<br/>+ HRR + CCR + VSR]
    PICK -->|Run all datasets| EVD[evaluate_datasets.py<br/>--only / --limit<br/>--skip-base / --skip-full]
    PICK -->|Calibrate EEDC| CAL[calibrate_eedc.py<br/>--data dev.jsonl]
    PICK -->|Build slides| BP[build_presentation.py]
    PICK -->|Prep TruthfulQA| BT[build_truthfulqa_jsonl.py]
    PICK -->|Launch UI| RF[run_frontend.py<br/>loads .env → Flask :5000]

    RI & RPT --> PIPE[Pipeline.run]
    EV & EVF & EVD --> PIPE
    BI --> EMBED[BGE-M3]
    EMBED --> IDX[FAISS + BM25 saved index]
    PIPE --> METRICS[(data/metrics_*.json)]
    EVF --> AGG[Detection aggregates<br/>ROC-AUC, HRR, CCR, VSR]
    CAL --> W[(data/eedc_weights.json)]
    W -.-> PIPE
    IDX -.-> PIPE
    METRICS -.-> FLASK_API[/api/evaluation/]

    classDef script fill:#0e7490,stroke:#67e8f9,color:#fff
    classDef io fill:#a16207,stroke:#fde047,color:#000
    classDef pipeBox fill:#1e3a8a,stroke:#60a5fa,color:#fff
    class RI,RPT,BI,EV,EVF,EVD,CAL,BP,BT,RF script
    class METRICS,W,IDX,AGG io
    class PIPE,EMBED pipeBox
```

---

## Layer 7 — Evaluation & Data Flow

```mermaid
flowchart LR
    subgraph DATASETS[Raw Datasets]
        D1[data/halueval_qa.jsonl<br/>21 MB]
        D2[data/truthfulqa_qa.jsonl<br/>1.1 MB]
        D3[Synthetic QA<br/>7 questions]
    end

    subgraph INDEX_BLD[Index Build]
        I1[chunk_documents<br/>size=400, overlap=50]
        I2[BGE-M3 Encode]
        I3[FAISS + BM25<br/>saved to disk]
    end

    subgraph EVAL[Evaluation Scripts]
        E1[evaluate.py<br/>BASE metrics]
        E2[evaluate_full.py<br/>+ detection + HRR/CCR/VSR]
        E3[scripts/evaluate_datasets.py<br/>orchestrator]
    end

    subgraph METRICS_OUT[Metrics Output]
        M1[data/metrics.json]
        M2[data/metrics_halueval.json]
        M3[data/metrics_truthfulqa.json]
    end

    subgraph FRONTEND_USE[Frontend Consumers]
        F1[/api/accuracy/]
        F2[/api/evaluation/comparison/]
        F3[/api/literature/*/]
    end

    subgraph LIT[Literature Material]
        L1[Paper/comparison_table.csv<br/>11 systems + Ours]
        L2[Paper/why_we_win.md<br/>verdict matrix + tally]
    end

    D1 & D2 & D3 --> E1
    D1 & D2 & D3 --> E2
    E1 --> M1
    E1 --> M2
    E1 --> M3
    E2 --> M2 & M3
    E2 --> M1
    E3 --> E1 & E2

    D1 & D2 --> I1
    I1 --> I2
    I2 --> I3
    I3 -.-> PIPE[Pipeline]

    M1 & M2 & M3 --> F1
    M1 & M2 & M3 --> F2
    L1 & L2 --> F3

    classDef data fill:#365314,stroke:#bef264,color:#fff
    classDef metric fill:#7f1d1d,stroke:#fca5a5,color:#fff
    classDef lit fill:#4c1d95,stroke:#c4b5fd,color:#fff
    classDef front fill:#0f766e,stroke:#5eead4,color:#fff
    class D1,D2,D3 data
    class M1,M2,M3 metric
    class L1,L2 lit
    class F1,F2,F3 front
```

---

## Layer 8 — Full End-to-End System View

This single diagram shows the entire stack from user input to evaluation storage.

```mermaid
flowchart TB
    %% ============ ENTRY ============
    USER([User]) -->|Query| ENTRY{Entry Point}

    ENTRY -->|Web UI :5000| FLASK[Flask Frontend<br/>frontend/app.py]
    ENTRY -->|REST :8000| FAST[FastAPI<br/>api/app.py]
    ENTRY -->|CLI| SCRIPT[scripts/run_inference.py<br/>or evaluate.py / report.py]
    ENTRY -->|Tests| TEST[tests/ — 31 pytest cases<br/>uses fake_pipeline fixture]

    %% ============ PIPELINE ============
    FLASK --> LAZY[Lazy Pipeline bootstrap]
    FAST --> ASYNC[asyncio.to_thread]
    SCRIPT --> DIRECT[Direct call]
    TEST --> FAKE[Fakes injected<br/>tests/_fakes.py]

    LAZY & ASYNC & DIRECT & FAKE --> PIPE[Pipeline<br/>src/pipeline.py]

    %% ============ 13 STEPS ============
    PIPE --> S1A[Step 1: Query Understanding]
    S1A --> S2A[Step 2: Grounded Generation]
    S2A --> S3A[Step 3: Early Detection]
    S3A --> S4A[Step 4: EEDC + Calibration]
    S4A --> S5A[Step 5: Claim Extraction]
    S5A --> S6A[Step 6: Hybrid Retrieval]
    S6A --> S7A[Step 7: Reranking]
    S7A --> S8A[Step 8: NLI Verification]
    S8A --> S9A[Step 9: Confidence Fusion]
    S9A --> S10A[Step 10: Regeneration]
    S10A --> S11A[Step 11: Final Gate]
    S11A --> S12A[Step 12: Explainable Response]
    S12A --> S13A[Step 13: Evaluation]

    %% ============ MODELS POOL ============
    subgraph MODELS[Model Pool — src/models/]
        MB[BGE-M3 Embedder]
        ML[Llama 3.1 Generator]
        MQ[Qwen Alt Generator]
        MO[OpenRouter]
        MR[BGE-Reranker-v2]
        MN[DeBERTa-v3-MNLI]
        MG[GLiNER]
        MS[spaCy]
        MX[XGBoost]
        ME[EEDC]
        MT[Temperature Scaler]
    end

    S1A & S2A & S3A & S4A & S5A & S6A & S7A & S8A & S9A & S10A --> MODELS

    %% ============ RETRIEVAL ============
    subgraph RET[Retrieval — src/retrieval/]
        FAISS[FAISSIndex<br/>IndexFlatIP]
        BM25[BM25Index<br/>pure-python]
        HYB[HybridRetriever<br/>RRF k=60]
    end
    S6A --> RET
    MB -.feeds.-> FAISS
    FAISS --> HYB
    BM25 --> HYB

    %% ============ UTILS ============
    subgraph UTILS[Utils — src/utils/]
        CHK[chunker.py]
        CLM[claim_extractor.py]
        DL[data_loader.py]
    end
    PIPE --> UTILS
    RET --> CHK
    S5A --> CLM
    SCRIPT --> DL

    %% ============ CONFIG ============
    CFG[configs/default.yaml<br/>+ HALLUC_* env vars<br/>src/config.py] -.configures.-> PIPE
    CFG -.configures.-> MODELS

    %% ============ OUTPUTS ============
    S12A --> RJSON[JSON Result]
    S12A --> RMD[Markdown Report]
    S13A --> METRICS[(data/metrics.json<br/>data/metrics_halueval.json<br/>data/metrics_truthfulqa.json)]

    RJSON --> ENTRY
    RMD --> ENTRY
    METRICS --> FLASK

    %% ============ EVAL + LIT ============
    subgraph EVAL_LAYER[Evaluation — src/evaluation/]
        RPT[report.py<br/>load_report + multi-dataset]
        LIT[literature.py<br/>Paper/comparison_table.csv + why_we_win.md]
    end
    METRICS --> RPT
    Paper[Paper/<br/>13 PDFs + comparison_table.csv<br/>+ why_we_win.md] --> LIT
    RPT --> FLASK
    LIT --> FLASK

    %% ============ STYLING ============
    classDef entry fill:#1e3a8a,stroke:#60a5fa,color:#fff
    classDef step fill:#0e7490,stroke:#67e8f9,color:#fff
    classDef model fill:#4c1d95,stroke:#c4b5fd,color:#fff
    classDef ret fill:#7c2d12,stroke:#fdba74,color:#fff
    classDef util fill:#365314,stroke:#bef264,color:#fff
    classDef io fill:#7f1d1d,stroke:#fca5a5,color:#fff
    classDef cfg fill:#525252,stroke:#d4d4d4,color:#fff

    class FLASK,FAST,SCRIPT,TEST,ENTRY entry
    class S1A,S2A,S3A,S4A,S5A,S6A,S7A,S8A,S9A,S10A,S11A,S12A,S13A,LAZY,ASYNC,DIRECT,FAKE step
    class MB,ML,MQ,MO,MR,MN,MG,MS,MX,ME,MT model
    class FAISS,BM25,HYB ret
    class CHK,CLM,DL util
    class METRICS,RJSON,RMD io
    class CFG cfg
```

---

## Layer 9 — Class / Module Dependency Graph (Code-Level)

For developers who want to trace `import` chains.

```mermaid
classDiagram
    class Pipeline {
        +cfg: Config
        +embedder: BGE_M3Embedder
        +generator: LlamaGenerator
        +generator_qwen: QwenGenerator
        +retriever: HybridRetriever
        +nli: DebertaNLIModel
        +reranker: BGEReranker
        +gliner_ner: GLiNERWrapper
        +spacy_ner: SpacyNER
        +early_classifier: XGBClassifier
        +eedc: EEDC
        +calibrator: TemperatureScaler
        +run(query) PipelineResult
        +build_index(docs)
        +load_index(dir)
    }

    class StepContext {
        +query: str
        +answer: str
        +claims: list~Claim~
        +evidence: list~Evidence~
        +feature_vector: dict
        +verdicts: list~ClaimVerdict~
        +eedc_score: float
        +confidence: float
        +decision: str
        +iterations: int
        +metadata: dict
    }

    class Config {
        +raw: dict
        +models() dict
        +prompt() dict
        +step(n) dict
    }

    class Step {
        <<abstract>>
        +number: int
        +name: str
        +run()
        +trace(payload, duration_ms)
    }

    class PipelineResult {
        +query: str
        +answer: str
        +claims: list~Claim~
        +evidence: list~Evidence~
        +confidence: float
        +decision: str
        +step_traces: list~StepTrace~
        +report_markdown: str
    }

    class Claim {
        +text: str
        +provenance: str
        +verdict: str
        +confidence: float
        +evidence: list~Evidence~
        +citation: str
    }

    class Evidence {
        +doc_id: str
        +text: str
        +score: float
    }

    class HybridRetriever {
        +faiss: FAISSIndex
        +bm25: BM25Index
        +rrf_k: float
        +add(doc_ids, docs, vecs)
        +search(query, qv, top_k) Evidence
    }

    Pipeline --> Config
    Pipeline --> HybridRetriever
    Pipeline --> StepContext
    Pipeline ..> Step : orchestrates
    StepContext --> Claim
    StepContext --> Evidence
    StepContext ..> PipelineResult : produces
    PipelineResult --> Claim
    PipelineResult --> Evidence
    PipelineResult --> StepTrace
    HybridRetriever --> FAISSIndex
    HybridRetriever --> BM25Index
    FAISSIndex --> Evidence
    BM25Index --> Evidence
```

---

## How to Render These Diagrams

| Tool | How to use |
|---|---|
| **GitHub** | Just open `architecture.md` in the repo — GitHub renders Mermaid blocks automatically |
| **VS Code** | Install the "Markdown Preview Mermaid Support" extension |
| **Obsidian** | Native Mermaid support — drop the file in your vault |
| **Mermaid Live Editor** | Paste any code block at <https://mermaid.live/> |
| **CLI** | `npx -p @mermaid-js/mermaid-cli mmdc -i architecture.md -o architecture.svg` |

---

## Diagram Index

1. **Layer 1** — The 13-step core pipeline (the heart of the project)
2. **Layer 2** — Models layer (12 HF wrappers + registry)
3. **Layer 3** — Retrieval layer (FAISS + BM25 + RRF)
4. **Layer 4** — API layer (FastAPI + Flask routes)
5. **Layer 5** — Frontend pages & JS modules
6. **Layer 6** — Scripts / CLI entry points
7. **Layer 7** — Evaluation & data flow
8. **Layer 8** — Full end-to-end system view
9. **Layer 9** — Class / module dependency graph

Every module, class, route, and script from the project is represented. Color coding:

- **Blue** — entry points & step pipeline
- **Cyan / teal** — pipeline internals
- **Purple** — models (HuggingFace)
- **Orange** — retrieval & decision points
- **Red** — I/O & metrics storage
- **Green** — utilities
- **Gray** — configuration
