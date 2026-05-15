# Legal Draft AI

> AI-powered document understanding, grounded drafting, and continuous improvement from operator edits — built as an assessment for **Pearson Specter Litt**.

---

## What It Does

Law firms deal with messy documents every day — scanned PDFs, handwritten notes, inconsistently formatted records. This system ingests those documents, pulls out usable information, generates grounded first-pass drafts, and gets better over time by learning from how operators edit those drafts.

```
Messy Input Documents
        ↓
┌──────────────────────┐
│  Document Processor  │  OCR + text extraction + field parsing
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Retrieval Layer     │  Chunking + embeddings + ChromaDB
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Draft Generator     │  Grounded drafts via LLM
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Edit Learner        │  Learns from operator edits → better future drafts
└──────────────────────┘
```

---

## Features

- **Handles messy inputs** — scanned PDFs, low-resolution images, handwritten notes, partially illegible records
- **OCR with preprocessing** — contrast enhancement, sharpening, resolution upscaling before Tesseract
- **Smart chunking** — paragraph-aware splitting with overlap to preserve legal clause context
- **Grounded generation** — every draft claim is traceable to a source chunk; unsupported claims are flagged as `[INSUFFICIENT SOURCE MATERIAL]`
- **Structured field extraction** — automatically pulls case numbers, dates, parties, amounts, courts
- **Five draft output types** — summary, case facts, internal memo, checklist, notice summary
- **Edit learning loop** — diffs operator edits, extracts style patterns, injects them into future prompts
- **Persistent memory** — learned preferences survive across sessions via JSON store
- **Zero API cost** — runs on Groq free tier + local sentence-transformers

---

## Project Structure

```
legal-draft-ai/
│
├── src/
│   ├── document_processor.py   # OCR, text extraction, field parsing
│   ├── retrieval.py            # Chunking, embeddings, ChromaDB vector store
│   ├── draft_generator.py      # LLM-powered grounded draft generation
│   ├── edit_learner.py         # Diff analysis + style pattern learning
│   └── pipeline.py             # Orchestrates all four stages
│
├── sample_inputs/              # Example messy legal documents
├── sample_outputs/             # Generated draft outputs
├── main.py                     # Full demo run
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/UttamAI-lab/legal-draft-ai.git
cd legal-draft-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

| OS | Command |
|----|---------|
| Windows | Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |
| Linux | `sudo apt install tesseract-ocr` |
| Mac | `brew install tesseract` |

### 4. Get a free Groq API key

1. Sign up at **https://console.groq.com** (free, no credit card)
2. Go to **API Keys → Create API Key**
3. Copy the key

### 5. Set up environment

```bash
# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

### 6. Run

```bash
python main.py
```

---

## Sample Output

Running `main.py` generates four draft files in `sample_outputs/`:

| File | Description |
|------|-------------|
| `case_facts.md` | Numbered facts grounded in source documents |
| `internal_memo.md` | To/From/Issue/Facts/Analysis/Conclusion format |
| `review_checklist.md` | Categorised YES/NO/PARTIAL checklist |
| `improved_memo.md` | Same memo type, improved after learning from operator edit |

**Example case fact (from sample run):**

```
3. The dispute relates to software development and maintenance services,
   as described in Schedule A of the Service Agreement.
   [Source: service_agreement.txt, Page: 1]
```

---

## Assumptions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| Groq free tier (LLaMA 3.3 70B) | No cost, high quality, sufficient rate limits for legal drafts |
| `sentence-transformers` for embeddings | Local, free, strong performance on domain text |
| ChromaDB as vector store | Zero-config, persistent, no external service needed |
| Paragraph-aware chunking with overlap | Preserves legal clause context better than fixed character splits |
| Regex-based field extraction | Fast, interpretable, sufficient for standard legal document patterns |
| Diff-based edit learning | Explainable, no fine-tuning infrastructure required |
| `temperature=0.3` for generation | Lower temperature reduces hallucination in factual legal drafts |

---

## Evaluation

| Component | What We Measure | Typical Result |
|-----------|----------------|----------------|
| Document Processing | Text yield, field extraction accuracy | 100% on digital text; ~80% on scanned docs |
| Retrieval | Relevance of top-5 chunks | 4–5 of 5 chunks directly relevant |
| Grounding Score | Content word overlap with source | 54–70% across draft types |
| Edit Learning | Patterns extracted per edit session | 1–3 patterns per operator edit |
| Draft Quality | Usability as first-pass output | Structured, sourced, ready for operator review |

---

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| OCR | Tesseract + PyMuPDF + pdfplumber | Free |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local |
| Vector Store | ChromaDB | Free, local |
| LLM | Groq API — LLaMA 3.3 70B | Free tier |
| CLI output | Rich | Free |

---

## Limitations

- Grounding score is a heuristic (word overlap), not semantic verification
- Edit learning works best after 3+ edits; single edits may produce noisy patterns
- Groq free tier has rate limits — large batch processing may need throttling
- OCR quality degrades on very low-resolution or handwritten documents

---

## License

MIT
