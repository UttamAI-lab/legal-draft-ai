"""
main.py
-------
Demonstrates the full pipeline with sample documents.
Run: python main.py
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import LegalDraftPipeline

console = Console()


def create_sample_documents():
    """Create sample messy legal documents for demonstration."""
    
    sample_dir = Path("sample_inputs")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample 1: Service Agreement (messy, inconsistent formatting)
    contract = """
SERVICE AGREEMENT
=================

    Case No: PSL-2024-0892

Between:
ALPHA TECH SOLUTIONS LTD ("Service Provider")
AND
BETA CORP INC ("Client")

Date: 15/03/2024

RE: Software Development  & Maintenance Services

1. SCOPE OF SERVICES

The Service Provider shall provide software development services
as described in Schedule A attached hereto.  

The client will pay the Service Provider a monthly fee of USD 15,000
for the duration of this Agreement.

2.  TERM

This Agreement shall commence on April 1, 2024 and shall continue
for a period of 12 months unless terminated earlier.

Termination: Either party may terminate with 30 days written notice.

3. CONFIDENTIALITY

Both parties agree to keep all information confidential.
The client's data must not be shared with third parties.

4. INTELLECTUAL PROPERTY

All work product created under this agreement shall be the property
of the Client upon full payment of all fees.

5. LIABILITY

The Service Provider's liability is limited to the total fees paid
in the preceding 3 months.

Signed:
Alpha Tech Solutions Ltd: _________________
Beta Corp Inc: _________________
"""

    # Sample 2: Legal Notice (partially unclear formatting)
    notice = """
LEGAL NOTICE
------------
Date: 22 March 2024

To: Beta Corp Inc
    123 Business Avenue
    New York, NY 10001

From:  Alpha Tech Solutions Ltd
       456 Tech Street, San Francisco, CA 94102

Subject: Notice of Breach of Contract - Case PSL-2024-0892

Dear Sir/Madam,

We write to formally notify you of your breach of the Service Agreement
dated 15/03/2024 (Case No: PSL-2024-0892).

Specific breaches identified:
1. Non-payment of March 2024 invoice (Amount: USD 15,000) due 01/04/2024
2. Unauthorized disclosure of proprietary source code to third party
3. Failure to provide access to systems as required under Schedule A

We hereby demand:
- Immediate payment of USD 15,000 overdue invoice
- Written explanation of the disclosure incident within 7 days
- Full system access restored within 48 hours

Failure to remedy these breaches within 14 days of this notice will result
in legal proceedings without further notice.

Yours faithfully,
[Signature]
Legal Department
Alpha Tech Solutions Ltd
"""

    # Sample 3: Internal notes (very messy)
    notes = """
Meeting Notes - client call 22/3
---------------------------------
talked to beta corp today -- they say they didnt get our invoice??
check with accounts - was invoice sent for march? amount should be $15k

they also mentioned something about code being shared -- need to 
investigate. john said he saw our code on github under different name??

next steps:
- send formal notice (already done)
- get evidence of github thing
- check if we have grounds to terminate

case number is PSL-2024-0892

potential damages: 15000 USD unpaid + potential IP claim
deadline for response from them: by April 5
"""

    (sample_dir / "service_agreement.txt").write_text(contract)
    (sample_dir / "legal_notice.txt").write_text(notice)
    (sample_dir / "meeting_notes.txt").write_text(notes)
    
    console.print("[green]✓ Sample documents created in sample_inputs/[/green]\n")
    return [
        str(sample_dir / "service_agreement.txt"),
        str(sample_dir / "legal_notice.txt"),
        str(sample_dir / "meeting_notes.txt")
    ]


def run_demo():
    console.print(Panel.fit(
        "[bold]Legal Draft AI — Pearson Specter Litt[/bold]\n"
        "Document Understanding + Grounded Drafting + Improvement from Edits",
        style="blue"
    ))
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]ERROR: GROQ_API_KEY not set in .env file[/red]")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = LegalDraftPipeline()
    
    # ─── STEP 1: Create and Ingest Documents ─────────────────────────
    console.print(Rule("[bold]STEP 1: Document Ingestion[/bold]"))
    
    doc_paths = create_sample_documents()
    
    for path in doc_paths:
        pipeline.ingest(path)
    
    # ─── STEP 2: Generate Various Draft Types ────────────────────────
    console.print(Rule("[bold]STEP 2: Draft Generation[/bold]"))
    
    output_dir = Path("sample_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 2a. Case Facts Summary
    console.print("\n[bold cyan]2a. Generating Case Facts Summary...[/bold cyan]")
    result_facts = pipeline.draft(
        draft_type="case_facts",
        task_description="Summarize the key facts of the dispute between Alpha Tech Solutions and Beta Corp",
        top_k=6
    )
    
    (output_dir / "case_facts.md").write_text(result_facts["draft"])
    console.print(result_facts["draft"])
    
    # 2b. Internal Memo
    console.print("\n[bold cyan]2b. Generating Internal Memo...[/bold cyan]")
    result_memo = pipeline.draft(
        draft_type="memo",
        task_description="Draft an internal memo assessing the breach of contract situation and recommended next steps",
        top_k=7
    )
    
    (output_dir / "internal_memo.md").write_text(result_memo["draft"])
    console.print(result_memo["draft"])
    
    # 2c. Document Checklist
    console.print("\n[bold cyan]2c. Generating Document Checklist...[/bold cyan]")
    result_checklist = pipeline.draft(
        draft_type="checklist",
        task_description="Create a checklist of items to verify before proceeding with legal action",
        top_k=5
    )
    
    (output_dir / "review_checklist.md").write_text(result_checklist["draft"])
    console.print(result_checklist["draft"])
    
    # ─── STEP 3: Simulate Operator Edit (Learning Loop) ──────────────
    console.print(Rule("[bold]STEP 3: Operator Edit & Learning[/bold]"))
    
    # Simulate: operator edits the memo to be more formal
    original_memo = result_memo["draft"]
    
    # A realistic operator edit: more formal language, concise
    edited_memo = original_memo.replace("will", "shall").replace(
        "looking into", "investigating"
    ).replace("get", "obtain")
    
    # Add a formal legal qualifier the operator inserted
    edited_memo = edited_memo + "\n\nWithout prejudice to any other rights or remedies available."
    
    learn_result = pipeline.record_edit(
        original=original_memo,
        edited=edited_memo,
        draft_type="memo"
    )
    
    # ─── STEP 4: Improved Draft (using learned patterns) ─────────────
    console.print(Rule("[bold]STEP 4: Improved Draft (Post-Learning)[/bold]"))
    
    console.print("[italic]Generating another memo — now with learned style preferences...[/italic]\n")
    
    result_improved = pipeline.draft(
        draft_type="memo",
        task_description="Draft a memo regarding the IP breach and recommended remedies",
        top_k=5
    )
    
    (output_dir / "improved_memo.md").write_text(result_improved["draft"])
    console.print(result_improved["draft"])
    
    # ─── Final Stats ─────────────────────────────────────────────────
    console.print(Rule("[bold]Pipeline Statistics[/bold]"))
    pipeline.show_stats()
    
    console.print(Panel(
        f"[bold green]✓ All outputs saved to sample_outputs/[/bold green]\n"
        f"  • case_facts.md\n"
        f"  • internal_memo.md\n"
        f"  • review_checklist.md\n"
        f"  • improved_memo.md",
        title="Done"
    ))


if __name__ == "__main__":
    run_demo()