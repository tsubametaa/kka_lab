"""Text normalization and training-candidate quality gate."""

import re
import unicodedata

from app.features.language.locales import (
    CLARIFICATION_MARKERS,
    STOP_WORDS,
    UNCERTAINTY_MARKERS,
    VAGUE_REFERENCES,
)

def normalize_question(question):
    normalized = unicodedata.normalize('NFKC', question).casefold()
    return ' '.join(re.findall(r'\w+', normalized, flags=re.UNICODE))

def _knowledge_terms(text):
    return {
        word for word in normalize_question(text).split()
        if len(word) >= 2 and word not in STOP_WORDS
    }

def training_quality(question, answer, consent):
    """Return (eligible, reason); eligibility still requires human review."""
    if not consent:
        return False, 'consent_not_given'

    normalized = normalize_question(question)
    words = normalized.split()
    if len(normalized) < 12 or len(words) < 3 or sum(c.isalnum() for c in normalized) < 8:
        return False, 'question_too_vague'

    content_terms = _knowledge_terms(question)
    if any(marker in normalized for marker in VAGUE_REFERENCES) and len(content_terms) < 3:
        return False, 'question_unclear'
    if len(content_terms) < 2:
        return False, 'question_unclear'

    if len(answer.strip()) < 40:
        return False, 'answer_too_short'

    normalized_answer = normalize_question(answer)
    if any(marker in normalized_answer for marker in CLARIFICATION_MARKERS):
        return False, 'assistant_requested_clarification'

    if any(marker in normalized_answer for marker in UNCERTAINTY_MARKERS):
        return False, 'assistant_uncertain'

    return True, 'pending_human_review'

