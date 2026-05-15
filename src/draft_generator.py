"""
Draft Generator
---------------
Uses Groq (free tier) to generate grounded draft outputs from:
- Retrieved evidence chunks
- Structured document fields  
- Learned style preferences from past edits

Model: llama-3.1-70b-versatile (free on Groq)
Output types: summary, memo, checklist, case_facts, notice_summary
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DRAFT_TYPE_CONFIGS = {
    "summary": {
        "description": "A concise title review or document summary",
        "format": "2-4 paragraphs, structured prose",
        "tone": "formal, objective"
    },
    "case_facts": {
        "description": "A case facts summary for internal legal review",
        "format": "Numbered facts, each grounded in source material",
        "tone": "precise, neutral, factual"
    },
    "memo": {
        "description": "A first-pass internal legal memorandum",
        "format": "To/From/Date/Re header, then Issue, Facts, Analysis, Conclusion",
        "tone": "formal legal memo style"
    },
    "checklist": {
        "description": "A document review checklist",
        "format": "Categorized checklist items with YES/NO/PARTIAL status",
        "tone": "concise, actionable"
    },
    "notice_summary": {
        "description": "A summary of a legal notice or correspondence",
        "format": "Key points extracted, parties identified, action items noted",
        "tone": "formal, clear"
    }
}


class DraftGenerator:
    """
    Generates grounded legal drafts using Groq (free tier).

    Grounding principle: every claim in the draft must be
    traceable to a retrieved source chunk. The system prompt
    enforces this explicitly.

    Free model used: llama-3.3-70b-versatile
    Rate limit (free tier): 6,000 tokens/min — sufficient for legal drafts.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = model

    def generate(
        self,
        draft_type: str,
        retrieved_chunks: list,
        structured_fields: dict,
        task_description: str,
        style_guidelines: str = "",
        extra_instructions: str = ""
    ) -> dict:
        """
        Generate a grounded draft.
        Returns dict with: draft text, grounding report, model used.
        """

        if draft_type not in DRAFT_TYPE_CONFIGS:
            draft_type = "summary"

        config = DRAFT_TYPE_CONFIGS[draft_type]

        evidence_block = self._format_evidence(retrieved_chunks)
        fields_block = self._format_fields(structured_fields)

        system_prompt = self._build_system_prompt(config, style_guidelines)
        user_message = self._build_user_message(
            task_description=task_description,
            draft_type=draft_type,
            config=config,
            evidence_block=evidence_block,
            fields_block=fields_block,
            extra_instructions=extra_instructions
        )

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,   # Low temp = more factual, less hallucination
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        draft_text = response.choices[0].message.content
        grounding_report = self._check_grounding(draft_text, retrieved_chunks)

        return {
            "draft": draft_text,
            "draft_type": draft_type,
            "grounding_report": grounding_report,
            "chunks_used": len(retrieved_chunks),
            "model": self.model,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        }

    def _build_system_prompt(self, config: dict, style_guidelines: str) -> str:
        return f"""You are a senior legal document analyst and drafter working for a law firm.

YOUR CORE RULE: Every statement in your draft MUST be grounded in the provided source evidence.
Do NOT add assumptions, inferences, or information not present in the documents.
If information is missing, say so explicitly rather than filling gaps.

OUTPUT TYPE: {config['description']}
FORMAT: {config['format']}
TONE: {config['tone']}

GROUNDING PROTOCOL:
- Only use facts explicitly stated in the EVIDENCE CHUNKS provided
- When citing a specific fact, note which chunk it came from (e.g., [Source: doc.pdf, p.2])
- If you cannot find sufficient evidence for a section, write [INSUFFICIENT SOURCE MATERIAL]
- Never write confident-sounding text built on unsupported assumptions

{style_guidelines}"""

    def _build_user_message(
        self, task_description, draft_type, config,
        evidence_block, fields_block, extra_instructions
    ) -> str:
        return f"""DRAFTING TASK: {task_description}

DRAFT TYPE REQUESTED: {draft_type.upper()}
Expected format: {config['format']}

--- EXTRACTED STRUCTURED FIELDS ---
{fields_block}

--- EVIDENCE CHUNKS (use ONLY these as your source) ---
{evidence_block}

--- INSTRUCTIONS ---
Generate a {draft_type} based strictly on the above evidence.
{extra_instructions}

Remember: This is a first-pass draft. Structure it well enough that an operator can
review and edit it. Do not fabricate any legal conclusions not supported by the source material."""

    def _format_evidence(self, chunks: list) -> str:
        """Format retrieved chunks into a readable evidence block."""
        if not chunks:
            return "[NO EVIDENCE RETRIEVED — check document ingestion]"

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            source_name = chunk.source.split('/')[-1]
            formatted.append(
                f"[CHUNK {i} | Source: {source_name} | Page: {chunk.page} | "
                f"Relevance: {chunk.relevance_score:.2f}]\n{chunk.text}"
            )

        return "\n\n---\n\n".join(formatted)

    def _format_fields(self, fields: dict) -> str:
        """Format structured fields for the prompt."""
        if not fields:
            return "No structured fields extracted."

        lines = []
        for key, values in fields.items():
            label = key.replace('_', ' ').title()
            lines.append(f"{label}: {', '.join(str(v) for v in values)}")

        return '\n'.join(lines)

    def _check_grounding(self, draft: str, chunks: list) -> dict:
        """
        Grounding check: how much of the draft can be traced
        back to source chunks by content word overlap.
        """
        if not chunks:
            return {"score": 0.0, "warning": "No source chunks available"}

        source_words = set()
        for chunk in chunks:
            source_words.update(chunk.text.lower().split())

        draft_words = draft.lower().split()
        if not draft_words:
            return {"score": 0.0}

        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'in', 'on', 'at', 'to',
            'for', 'of', 'and', 'or', 'but', 'this', 'that', 'with'
        }

        content_words = [
            w for w in draft_words
            if w not in stopwords and len(w) > 3
        ]
        grounded = [w for w in content_words if w in source_words]
        score = len(grounded) / max(len(content_words), 1)

        return {
            "score": round(score, 2),
            "grounded_words": len(grounded),
            "total_content_words": len(content_words),
            "assessment": (
                "Well-grounded" if score > 0.6
                else "Partially grounded" if score > 0.35
                else "Weakly grounded — review carefully"
            )
        }