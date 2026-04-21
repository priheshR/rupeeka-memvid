import os
import json
import re
from typing import Generator, Optional
from google import genai


def _build_extraction_prompt(text: str) -> str:
    return (
        "You are a professional fact-checker. Analyse the following text and identify every distinct claim it contains.\n\n"
        "For each claim:\n"
        "1. Extract the exact claim as a direct quote from the text\n"
        "2. Classify it as one of:\n"
        "   - UNVERIFIABLE: opinion, prediction, value judgment, normative statement\n"
        "   - MIXED: has a verifiable factual component wrapped in interpretation\n"
        "   - VERIFIABLE: specific factual assertion that can be confirmed or refuted\n\n"
        "For MIXED claims also extract:\n"
        "   - factual_component: the part that can be checked\n"
        "   - interpretive_layer: the opinion/interpretation layer\n\n"
        "Respond ONLY with a valid JSON array, no markdown, no extra text:\n"
        "[\n"
        "  {\n"
        '    "claim": "<exact quote>",\n'
        '    "classification": "VERIFIABLE or MIXED or UNVERIFIABLE",\n'
        '    "factual_component": "<only for MIXED>",\n'
        '    "interpretive_layer": "<only for MIXED>",\n'
        '    "unverifiable_reason": "<only for UNVERIFIABLE>"\n'
        "  }\n"
        "]\n\n"
        "Text to analyse:\n"
        '"""\n'
        + text[:4000]
        + '\n"""'
    )


def _build_scoring_prompt(claim: dict, source: str, kb_context: str) -> str:
    factual_note = ""
    if claim.get("classification") == "MIXED":
        factual_note = (
            "FACTUAL COMPONENT: " + claim.get("factual_component", "") + "\n"
            "INTERPRETIVE LAYER: " + claim.get("interpretive_layer", "") + "\n\n"
        )

    return (
        "You are a professional fact-checker using Tree of Thought reasoning.\n\n"
        "CLAIM: " + claim["claim"] + "\n"
        "CLASSIFICATION: " + claim["classification"] + "\n"
        + factual_note
        + "SOURCE (if provided): " + (source or "not provided") + "\n\n"
        "KNOWLEDGE BASE CONTEXT:\n"
        '"""\n' + kb_context + '\n"""\n\n'
        "Work through all four branches:\n\n"
        "BRANCH 1 — THE ADVOCATE:\n"
        "What is the strongest case that this claim is accurate and fairly presented?\n\n"
        "BRANCH 2 — THE CRITIC:\n"
        "What is the strongest case this claim is misleading or false?\n\n"
        "BRANCH 3 — THE METHODOLOGIST:\n"
        "Is the underlying evidence of sufficient quality? Consider methodology, conflicts of interest, peer review.\n\n"
        "BRANCH 4 — THE CONTEXTUALISER:\n"
        "Is this claim applied in the right context? Consider temporal validity, geography, population match.\n\n"
        "Then score each criterion:\n"
        "1. source_quality (0-3 stars): credible, authoritative, current source?\n"
        "2. accurate_representation (0-3 stars): does claim reflect what source says?\n"
        "3. contextual_completeness (0-2 stars): are counter-findings omitted?\n"
        "4. evidence_claim_alignment (0-2 stars): does evidence logically support claim?\n"
        "5. language_calibration (0-2 stars): is language proportional to evidence strength?\n"
        "6. independent_corroboration (0-3 stars): confirmed through independent sources?\n\n"
        "Severity flags — include any that apply:\n"
        "OUTDATED, MISLEADING_BY_OMISSION, OVERSTATED, MISATTRIBUTED, CONTRADICTED, UNSUBSTANTIATED\n\n"
        "Confidence: HIGH, MEDIUM, or LOW\n\n"
        "Respond ONLY with valid JSON, no markdown:\n"
        "{\n"
        '  "branch_advocate": "<2-3 sentences>",\n'
        '  "branch_critic": "<2-3 sentences>",\n'
        '  "branch_methodologist": "<2-3 sentences>",\n'
        '  "branch_contextualiser": "<2-3 sentences>",\n'
        '  "criteria": {\n'
        '    "source_quality":            {"stars": 0, "max": 3, "reasoning": "<one sentence>"},\n'
        '    "accurate_representation":   {"stars": 0, "max": 3, "reasoning": "<one sentence>"},\n'
        '    "contextual_completeness":   {"stars": 0, "max": 2, "reasoning": "<one sentence>"},\n'
        '    "evidence_claim_alignment":  {"stars": 0, "max": 2, "reasoning": "<one sentence>"},\n'
        '    "language_calibration":      {"stars": 0, "max": 2, "reasoning": "<one sentence>"},\n'
        '    "independent_corroboration": {"stars": 0, "max": 3, "reasoning": "<one sentence>"}\n'
        "  },\n"
        '  "severity_flags": [],\n'
        '  "confidence": "HIGH or MEDIUM or LOW",\n'
        '  "confidence_reason": "<one sentence>",\n'
        '  "summary": "<2-3 sentence plain language synthesis>"\n'
        "}"
    )


class FactChecker:
    """Claim-based fact checker using Tree of Thought reasoning.

    Streams results claim by claim so the UI can display
    each result as it arrives rather than waiting for all claims.
    """

    def __init__(self, kb=None, model: str = "gemini-2.5-pro"):
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model
        self.kb = kb

    def _call_gemini(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()

    def _parse_json(self, text: str) -> any:
        """Extract and parse JSON from Gemini response."""
        # Strip markdown fences
        cleaned = re.sub(r'```(?:json)?\s*', '', text).strip()
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Find outermost array
        start = cleaned.find('[')
        if start != -1:
            end = cleaned.rfind(']')
            if end != -1:
                try:
                    return json.loads(cleaned[start:end + 1])
                except Exception:
                    pass

        # Find outermost object
        start = cleaned.find('{')
        if start != -1:
            end = cleaned.rfind('}')
            if end != -1:
                try:
                    return json.loads(cleaned[start:end + 1])
                except Exception:
                    pass

        raise ValueError(f"Could not parse JSON: {text[:300]}")

    def _extract_claims(self, text: str) -> list:
        prompt = _build_extraction_prompt(text)
        raw = self._call_gemini(prompt)
        return self._parse_json(raw)

    def _get_kb_context(self, claim_text: str) -> str:
        if self.kb is None:
            return "No knowledge base available."
        try:
            results = self.kb.search(claim_text, top_k=3, pipeline='fast')
            if not results:
                return "No relevant content found in knowledge base."
            return '\n\n'.join([r['text'] for r in results])
        except Exception:
            return "Knowledge base search unavailable."

    def _score_claim(self, claim: dict, source: str = "") -> dict:
        kb_context = self._get_kb_context(
            claim.get('factual_component', claim['claim'])
        )
        prompt = _build_scoring_prompt(claim, source, kb_context)
        raw = self._call_gemini(prompt)
        scoring = self._parse_json(raw)

        # Calculate totals — coerce to int in case Gemini returns strings
        criteria = scoring.get('criteria', {})
        total = sum(int(v.get('stars', 0)) for v in criteria.values())
        max_total = sum(int(v.get('max', 0)) for v in criteria.values())
        
        # Also normalise stars in criteria so UI renders correctly
        for k in criteria:
            criteria[k]['stars'] = int(criteria[k].get('stars', 0))
            criteria[k]['max']   = int(criteria[k].get('max', 0))

        return {
            **claim,
            **scoring,
            'total_stars': total,
            'max_stars': max_total,
            'kb_context_used': 'No relevant' not in kb_context,
        }

    def analyse_stream(
        self,
        text: str,
        source: str = "",
    ) -> Generator[str, None, None]:
        """Analyse text and stream results claim by claim as SSE events."""

        def sse(event: str, data: dict) -> str:
            return "event: " + event + "\ndata: " + json.dumps(data) + "\n\n"

        yield sse("status", {"message": "Identifying claims...", "step": 1})

        try:
            claims = self._extract_claims(text)
        except Exception as e:
            yield sse("error", {"message": "Could not extract claims: " + str(e)})
            return

        total_claims = len(claims)
        verifiable_count = sum(
            1 for c in claims if c['classification'] in ['VERIFIABLE', 'MIXED']
        )

        yield sse("claims_found", {
            "total": total_claims,
            "verifiable": verifiable_count,
            "unverifiable": total_claims - verifiable_count,
            "claims_preview": [c['claim'][:80] for c in claims],
        })

        for i, claim in enumerate(claims):
            yield sse("claim_start", {
                "index": i,
                "total": total_claims,
                "claim": claim['claim'],
                "classification": claim['classification'],
                "message": "Analysing claim " + str(i + 1) + " of " + str(total_claims) + "...",
            })

            if claim['classification'] == 'UNVERIFIABLE':
                yield sse("claim_result", {
                    "index": i,
                    "claim": claim['claim'],
                    "classification": "UNVERIFIABLE",
                    "unverifiable_reason": claim.get('unverifiable_reason', ''),
                    "total_stars": None,
                    "max_stars": None,
                })
                continue

            try:
                result = self._score_claim(claim, source)
                yield sse("claim_result", {"index": i, **result})
            except Exception as e:
                yield sse("claim_error", {
                    "index": i,
                    "claim": claim['claim'],
                    "error": str(e),
                })

        yield sse("complete", {
            "total_claims": total_claims,
            "verifiable_analysed": verifiable_count,
            "message": "Analysis complete.",
        })


# Singleton
_factchecker = None

def get_factchecker(kb=None):
    global _factchecker
    if _factchecker is None:
        _factchecker = FactChecker(kb=kb)
    elif kb is not None and _factchecker.kb is None:
        _factchecker.kb = kb
    return _factchecker
