"""Step 0 of QNPruner: text parsing (spaCy ``en_core_web_sm``, no word list, no LLM).

    split_text(text)      question / context split by rules (SQA3D "situation + question" prompts)
    question_nouns(text)  unique NOUN / PROPN surfaces of the question sentence, first-appearance order
    attribute_phrases()   amod noun phrases (NPG1A): "root + compound + amod" strings of noun chunks with an amod
    question_type(text)   QR3 type A / B / C that sets the Stage-1 share rho of the budget
When the question sentence cannot be isolated (split_text -> ambiguous), the whole prompt is parsed instead.
"""
import re

NOUN_POS = ('NOUN', 'PROPN')
WH_WORDS = {'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how'}
CORE_DEPS = ('nsubj', 'nsubjpass', 'attr', 'dobj')
RHO = {'A': 20 / 32, 'B': 12 / 32, 'C': 1 / 2}          # Stage-1 share of the budget per question type
_MARKER = re.compile(r'(?i)\b(situation|context|question)\s*:')
_NLP = {}


def nlp():
    if 'nlp' not in _NLP:
        import spacy
        _NLP['nlp'] = spacy.load('en_core_web_sm')
    return _NLP['nlp']


_DOCS = {}


def parse(s):
    """spaCy parse of s, cached per string: one question's split / nouns / phrases / type re-read the same span, and the
    parse is deterministic, so a cached Doc gives the same result as a fresh one (the Docs are only read)."""
    doc = _DOCS.get(s)
    if doc is None:
        if len(_DOCS) >= 64:
            _DOCS.clear()
        doc = _DOCS[s] = nlp()(s)
    return doc


# ------------------------------------------------------------------------------------------------ question / context
def _is_question(sent):
    toks = [t for t in sent if not t.is_space and not t.is_punct]
    if sent.text.rstrip().endswith('?'):
        return True
    if toks and toks[0].lower_ in WH_WORDS:
        return True
    return len(toks) > 1 and toks[0].pos_ == 'AUX' and toks[1].pos_ in ('PRON', 'NOUN', 'PROPN', 'DET')


def split_text(text):
    """{'text', 'question_span', 'context_span', 'ambiguous'}. Explicit "Situation:/Question:" markers first; else the
    question sentence must be the last sentence and the only one (wh-first, '?', or aux-subject inversion); anything
    else is ambiguous (the whole text is used)."""
    out = {'text': text, 'question_span': None, 'context_span': None, 'ambiguous': False}
    marks = list(_MARKER.finditer(text))
    if marks:
        parts = {}
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            key = 'question' if m.group(1).lower() == 'question' else 'context'
            if key in parts:
                return dict(out, ambiguous=True)
            seg = text[m.end():end]
            s = m.end() + len(seg) - len(seg.lstrip())
            parts[key] = (s, s + len(seg.strip()))
        if 'question' not in parts:
            return dict(out, ambiguous=True)
        out['question_span'] = list(parts['question'])
        if parts.get('context'):
            out['context_span'] = list(parts['context'])
        return out
    sents = [s for s in parse(text).sents if s.text.strip()]
    qs = [i for i, s in enumerate(sents) if _is_question(s)]
    if len(qs) != 1 or qs[0] != len(sents) - 1:
        return dict(out, ambiguous=True)
    q = sents[qs[0]]
    out['question_span'] = [q.start_char, q.start_char + len(q.text.strip())]
    if qs[0] > 0:
        c0, c1 = sents[0].start_char, sents[qs[0] - 1].end_char
        out['context_span'] = [c0, c0 + len(text[c0:c1].strip())]
    return out


def _span(split):
    return [0, len(split['text'])] if split['ambiguous'] else split['question_span']


# ------------------------------------------------------------------------------------------------ nouns / phrases
def question_nouns(split):
    """Unique (lower-case) NOUN / PROPN surfaces in first-appearance order. A noun counts when it occurs in the question
    sentence; its surface is that of its first occurrence in the prompt (context included). Ambiguous -> whole text."""
    text = split['text']

    def nouns_in(span, source):
        doc = parse(text[span[0]:span[1]])
        return [(span[0] + t.idx, t.text, source) for t in doc if t.pos_ in NOUN_POS]

    if split['ambiguous']:
        raw = nouns_in((0, len(text)), 'text')
    else:
        raw = nouns_in(split['context_span'], 'context') if split['context_span'] else []
        raw += nouns_in(split['question_span'], 'question')
        raw.sort(key=lambda r: r[0])
    first, sources = {}, {}
    for _, surface, source in raw:
        key = surface.lower()
        first.setdefault(key, surface)
        sources.setdefault(key, set()).add(source)
    keep = [k for k in first if split['ambiguous'] or 'question' in sources[k]]
    return [first[k] for k in keep]


def _modifier_tokens(chunk):
    """Tokens of the chunk reachable from its root through compound / amod arcs, in text order."""
    keep, stack = {chunk.root.i}, [chunk.root]
    while stack:
        h = stack.pop()
        for c in h.children:
            if chunk.start <= c.i < chunk.end and c.dep_ in ('compound', 'amod') and c.i not in keep:
                keep.add(c.i)
                stack.append(c)
    return [t for t in chunk if t.i in keep]


def attribute_phrases(split, nouns):
    """[(phrase, head noun lower-case)]: question noun chunks (NOUN / PROPN root, unique by lower-cased chunk text) with
    an amod token; phrase = root + compound + amod tokens; kept if not equal to a question noun and not seen before."""
    if not nouns:
        return []
    s0, s1 = _span(split)
    doc = parse(split['text'][s0:s1])
    have = {n.lower() for n in nouns}
    seen, out = set(), []
    for ch in doc.noun_chunks:
        if ch.root.pos_ not in NOUN_POS or ch.text.lower() in seen:
            continue
        seen.add(ch.text.lower())
        if not any(t.dep_ == 'amod' for t in ch):
            continue
        phrase = ' '.join(t.text for t in _modifier_tokens(ch))
        if phrase.strip() and phrase.lower() not in have and phrase not in [p for p, _ in out]:
            out.append((phrase, ch.root.text.lower()))
    return out


def question_type(split):
    """QR3: no noun chunk -> C; a noun chunk in a core slot (nsubj / nsubjpass / attr / dobj) -> A; a WP wh-pronoun in a
    core slot -> B; else C."""
    s0, s1 = _span(split)
    doc = parse(split['text'][s0:s1])
    chunks = [c for c in doc.noun_chunks if c.root.pos_ in NOUN_POS]
    if not chunks:
        return 'C'
    if any(c.root.dep_ in CORE_DEPS for c in chunks):
        return 'A'
    if any(t.tag_ == 'WP' and t.dep_ in CORE_DEPS for t in doc):
        return 'B'
    return 'C'
