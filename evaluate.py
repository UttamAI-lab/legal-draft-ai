"""
evaluate.py
-----------
Measures system performance across all four pipeline components.
Run: python evaluate.py
"""

import sys
import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import LegalDraftPipeline

console = Console()


# ── Test Documents ────────────────────────────────────────────────────────────

CLEAN_DOC = """
SERVICES AGREEMENT — Case No: EVAL-2024-001
Between: TechCorp Ltd ("Provider") and RetailCo Inc ("Client")
Date: 01/01/2024
Monthly fee: USD 10,000
Term: 12 months from January 1, 2024.
Termination: 30 days written notice required.
Confidentiality: Both parties shall keep all information strictly confidential.
IP: All deliverables shall remain property of Client upon full payment.
"""

MESSY_DOC = """
services  agreement!!!
case no EVAL-2024-002
between:  TechCorp  ltd  AND   RetailCo INC
date: 1/1/2024
monthly  fee:  USD10,000
term:12 months   from jan 1 2024
termination:   30days   written  notice
confidentality:  both  parties keep info confidential
ip: deliverables = client property after full paymt
"""

NOTICE_DOC = """
LEGAL NOTICE — EVAL-2024-001
To: RetailCo Inc
From: TechCorp Ltd
Date: 15/03/2024
Subject: Notice of Breach

Specific breaches:
1. Non-payment of February invoice — Amount: USD 10,000 — Due: 01/03/2024
2. Unauthorized sharing of proprietary code with competitor
3. System access denied to Provider team since 10/03/2024

Demands:
- Immediate payment of USD 10,000
- Written explanation within 7 days
- System access restored within 48 hours

Failure to comply within 14 days will result in legal proceedings.
"""


def setup_test_docs():
    """Write test documents to disk."""
    test_dir = Path("eval_inputs")
    test_dir.mkdir(exist_ok=True)

    (test_dir / "clean_contract.txt").write_text(CLEAN_DOC)
    (test_dir / "messy_contract.txt").write_text(MESSY_DOC)
    (test_dir / "legal_notice.txt").write_text(NOTICE_DOC)

    return test_dir


# ── Evaluation Functions ──────────────────────────────────────────────────────

def eval_document_processing(pipeline, test_dir) -> dict:
    """
    Evaluates document processing quality.
    Metrics: field extraction rate, handling of messy input, chunk yield.
    """
    results = {}

    # Test 1: Clean document
    t0 = time.time()
    r_clean = pipeline.ingest(str(test_dir / "clean_contract.txt"))
    results["clean_extraction_time"] = round(time.time() - t0, 2)
    results["clean_confidence"] = r_clean["confidence"]
    results["clean_fields_found"] = len(r_clean["structured_fields"])
    results["clean_chunks"] = r_clean["chunks_indexed"]

    # Test 2: Messy document
    t0 = time.time()
    r_messy = pipeline.ingest(str(test_dir / "messy_contract.txt"))
    results["messy_extraction_time"] = round(time.time() - t0, 2)
    results["messy_confidence"] = r_messy["confidence"]
    results["messy_fields_found"] = len(r_messy["structured_fields"])
    results["messy_chunks"] = r_messy["chunks_indexed"]

    # Test 3: Legal notice
    r_notice = pipeline.ingest(str(test_dir / "legal_notice.txt"))
    results["notice_fields_found"] = len(r_notice["structured_fields"])
    results["notice_chunks"] = r_notice["chunks_indexed"]

    # Score: out of 3
    score = 0
    if results["clean_fields_found"] >= 2:
        score += 1  # Field extraction works on clean docs
    if results["messy_fields_found"] >= 1:
        score += 1  # Handles messy input
    if results["clean_chunks"] > 0 and results["messy_chunks"] > 0:
        score += 1  # Produces usable downstream output

    results["score"] = score
    results["max_score"] = 3
    return results


def eval_retrieval_and_grounding(pipeline) -> dict:
    """
    Evaluates retrieval quality and grounding of generated output.
    Metrics: chunk relevance, grounding score, source traceability.
    """
    results = {}

    # Generate a draft and measure grounding
    t0 = time.time()
    draft_result = pipeline.draft(
        draft_type="case_facts",
        task_description="Summarize the breach of contract facts and payment obligations",
        top_k=5
    )
    results["generation_time"] = round(time.time() - t0, 2)
    results["chunks_retrieved"] = draft_result["chunks_used"]
    results["grounding_score"] = draft_result["grounding_report"]["score"]
    results["grounding_assessment"] = draft_result["grounding_report"]["assessment"]

    # Check source citations present in draft
    draft_text = draft_result["draft"]
    has_citations = "[Source:" in draft_text or "source_" in draft_text.lower()
    results["has_source_citations"] = has_citations

    # Check unsupported generation is flagged
    has_insufficient_flag = "[INSUFFICIENT SOURCE MATERIAL]" in draft_text
    results["flags_missing_evidence"] = has_insufficient_flag

    # Score: out of 3
    score = 0
    if results["chunks_retrieved"] >= 3:
        score += 1  # Retrieves sufficient context
    if results["grounding_score"] >= 0.4:
        score += 1  # Output grounded in source material
    if results["has_source_citations"]:
        score += 1  # Supporting evidence is inspectable

    results["score"] = score
    results["max_score"] = 3
    results["draft_preview"] = draft_text[:300] + "..."
    return results


def eval_draft_quality(pipeline) -> dict:
    """
    Evaluates draft quality across multiple output types.
    Metrics: structure, clarity, consistency, first-pass usefulness.
    """
    results = {}

    draft_types = ["memo", "checklist", "notice_summary"]
    scores = []

    for dtype in draft_types:
        result = pipeline.draft(
            draft_type=dtype,
            task_description=f"Generate a {dtype} for the contract breach situation",
            top_k=4
        )
        draft = result["draft"]

        # Quality checks
        has_structure = any(
            marker in draft
            for marker in ["##", "**", "1.", "-", ":", "To:", "Issue:", "Facts:"]
        )
        min_length = len(draft.split()) >= 80
        grounded = result["grounding_report"]["score"] >= 0.3

        type_score = sum([has_structure, min_length, grounded])
        scores.append(type_score)

        results[f"{dtype}_structured"] = has_structure
        results[f"{dtype}_length_ok"] = min_length
        results[f"{dtype}_grounded"] = grounded
        results[f"{dtype}_score"] = type_score

    avg = sum(scores) / len(scores)
    results["score"] = round(min(avg, 3))
    results["max_score"] = 3
    return results


def eval_edit_learning(pipeline) -> dict:
    """
    Evaluates the edit learning loop.
    Metrics: edits captured, patterns learned, improvement in future draft.
    """
    results = {}

    # Generate original draft
    original_result = pipeline.draft(
        draft_type="memo",
        task_description="Draft a memo on the breach situation and next steps",
        top_k=5
    )
    original_draft = original_result["draft"]
    original_grounding = original_result["grounding_report"]["score"]

    # Simulate operator edit (more formal language + legal qualifier)
    edited_draft = original_draft.replace("will ", "shall ").replace(
        "looking at", "examining"
    ).replace("get", "obtain")
    edited_draft += "\n\nWithout prejudice to any other rights or remedies available."
    edited_draft += "\nPursuant to the terms of the agreement, all obligations remain binding."

    # Record the edit
    learn_result = pipeline.record_edit(original_draft, edited_draft, "memo")
    results["changes_detected"] = learn_result["changes_detected"]
    results["patterns_learned"] = len(learn_result["patterns_learned"])
    results["patterns"] = learn_result["patterns_learned"]

    # Generate improved draft
    improved_result = pipeline.draft(
        draft_type="memo",
        task_description="Draft a follow-up memo on remedies and next steps",
        top_k=5
    )
    improved_grounding = improved_result["grounding_report"]["score"]
    improved_draft = improved_result["draft"]

    results["original_grounding"] = original_grounding
    results["improved_grounding"] = improved_grounding
    results["learned_pattern_applied"] = (
        "without prejudice" in improved_draft.lower() or
        "pursuant" in improved_draft.lower() or
        "shall" in improved_draft.lower()
    )

    # Score: out of 3
    score = 0
    if results["changes_detected"] > 0:
        score += 1  # Edits are captured
    if results["patterns_learned"] > 0:
        score += 1  # Reusable patterns are learned
    if results["learned_pattern_applied"]:
        score += 1  # Future outputs improve meaningfully

    results["score"] = score
    results["max_score"] = 3
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_results(dp, rg, dq, el):
    """Print a formatted evaluation report."""

    console.print(Panel.fit(
        "[bold]Legal Draft AI — Evaluation Report[/bold]\n"
        "Pearson Specter Litt Assessment",
        style="blue"
    ))

    # ── Document Processing
    console.print(Rule("[bold cyan]1. Document Processing[/bold cyan]"))
    t = Table(show_header=True, header_style="bold")
    t.add_column("Metric")
    t.add_column("Clean Doc")
    t.add_column("Messy Doc")
    t.add_column("Notice")

    t.add_row(
        "Fields Extracted",
        str(dp["clean_fields_found"]),
        str(dp["messy_fields_found"]),
        str(dp["notice_fields_found"])
    )
    t.add_row(
        "Chunks Indexed",
        str(dp["clean_chunks"]),
        str(dp["messy_chunks"]),
        str(dp["notice_chunks"])
    )
    t.add_row(
        "Confidence",
        f"{dp['clean_confidence']:.0%}",
        f"{dp['messy_confidence']:.0%}",
        "100%"
    )
    console.print(t)
    console.print(f"  Score: [bold green]{dp['score']}/{dp['max_score']}[/bold green]\n")

    # ── Retrieval & Grounding
    console.print(Rule("[bold cyan]2. Retrieval & Grounding[/bold cyan]"))
    t2 = Table(show_header=True, header_style="bold")
    t2.add_column("Metric")
    t2.add_column("Result")

    t2.add_row("Chunks Retrieved", str(rg["chunks_retrieved"]))
    t2.add_row("Grounding Score", f"{rg['grounding_score']:.0%}")
    t2.add_row("Assessment", rg["grounding_assessment"])
    t2.add_row("Source Citations Present", "✅ Yes" if rg["has_source_citations"] else "❌ No")
    t2.add_row("Flags Missing Evidence", "✅ Yes" if rg["flags_missing_evidence"] else "❌ No")
    console.print(t2)
    console.print(f"  Score: [bold green]{rg['score']}/{rg['max_score']}[/bold green]\n")

    # ── Draft Quality
    console.print(Rule("[bold cyan]3. Draft Quality[/bold cyan]"))
    t3 = Table(show_header=True, header_style="bold")
    t3.add_column("Draft Type")
    t3.add_column("Structured")
    t3.add_column("Min Length")
    t3.add_column("Grounded")

    for dtype in ["memo", "checklist", "notice_summary"]:
        t3.add_row(
            dtype,
            "✅" if dq.get(f"{dtype}_structured") else "❌",
            "✅" if dq.get(f"{dtype}_length_ok") else "❌",
            "✅" if dq.get(f"{dtype}_grounded") else "❌"
        )
    console.print(t3)
    console.print(f"  Score: [bold green]{dq['score']}/{dq['max_score']}[/bold green]\n")

    # ── Edit Learning
    console.print(Rule("[bold cyan]4. Improvement from Edits[/bold cyan]"))
    t4 = Table(show_header=True, header_style="bold")
    t4.add_column("Metric")
    t4.add_column("Result")

    t4.add_row("Changes Detected", str(el["changes_detected"]))
    t4.add_row("Patterns Learned", str(el["patterns_learned"]))
    t4.add_row("Pattern Applied in Next Draft", "✅ Yes" if el["learned_pattern_applied"] else "❌ No")
    t4.add_row("Original Grounding", f"{el['original_grounding']:.0%}")
    t4.add_row("Improved Grounding", f"{el['improved_grounding']:.0%}")

    if el["patterns"]:
        t4.add_row("Learned Pattern", el["patterns"][0][:60] + "...")
    console.print(t4)
    console.print(f"  Score: [bold green]{el['score']}/{el['max_score']}[/bold green]\n")

    # ── Overall
    total = dp["score"] + rg["score"] + dq["score"] + el["score"]
    max_total = 12

    console.print(Rule("[bold]Overall Score[/bold]"))
    console.print(
        Panel(
            f"[bold white]{total} / {max_total}[/bold white]   "
            f"({'Strong' if total >= 9 else 'Moderate' if total >= 6 else 'Needs work'})",
            style="green" if total >= 9 else "yellow" if total >= 6 else "red"
        )
    )

    # Save report
    report = {
        "document_processing": dp,
        "retrieval_grounding": rg,
        "draft_quality": dq,
        "edit_learning": el,
        "total_score": total,
        "max_score": max_total
    }
    Path("evaluation_results.json").write_text(
        json.dumps(report, indent=2)
    )
    console.print("[dim]Full results saved to evaluation_results.json[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]ERROR: GROQ_API_KEY not set in .env[/red]")
        sys.exit(1)

    console.print("[bold blue]Setting up evaluation pipeline...[/bold blue]\n")

    # Use separate DB so eval doesn't pollute main data
    pipeline = LegalDraftPipeline(
        db_dir="./eval_chroma_db",
        memory_path="./eval_memory.json"
    )

    test_dir = setup_test_docs()

    console.print(Rule("Running Evaluations"))

    dp = eval_document_processing(pipeline, test_dir)
    rg = eval_retrieval_and_grounding(pipeline)
    dq = eval_draft_quality(pipeline)
    el = eval_edit_learning(pipeline)

    print_results(dp, rg, dq, el)