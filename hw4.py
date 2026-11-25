"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

import math
from collections import defaultdict

from bs4 import BeautifulSoup
from bs4.element import NavigableString

# System switches. The defaults are the final system described in README.md; ablation.py flips them
# one at a time to produce the Task 5 table.
CONFIG = {
    "table_to_text": True,        # V1: structured table text instead of a cell stream
    "structured_chunking": True,  # V2: one chunk per table, narrative grouped by section
    "metadata_filter": True,      # V3: candidate pool = the question's company and document type
    "hybrid_scoring": True,       # V4: year bonus + IDF coverage + table-row bonus on top of BM25
    "context_header": True,       # V5: a metadata + section + caption header line on every chunk
    "better_tokenizer": True,     # V6: keep numbers and years, fold plurals, expand synonyms
    "per_company_retrieval": True,  # V7: one ranking per company for comparisons
    "abstention": True,           # V8: return [] when the corpus cannot answer
    "compact_context": True,      # V9: dedupe, cap at 3-4 chunks, drop the tail below 35%
}

_MAX_FINAL = 4                # chunks handed to the generator when compact_context is on
_NAIVE_CHUNK_CHARS = 1000     # B0/V1: RecursiveCharacterTextSplitter chunk size
_NAIVE_CHUNK_OVERLAP = 150    # B0/V1: its overlap

_MAX_BLOCK_CHARS = 1200       # narrative blocks are grouped up to this size
_TABLE_CHUNK_CHARS = 1300     # one table is split on row boundaries beyond this
_PREFERRED_ROWS = 3           # chunks per company for a single-company question
_PREFERRED_ROWS_CMP = 2       # chunks per company for a comparison
_SCORE_KEEP = 0.35            # drop candidates far below the best one (compact_context only)
_COVERAGE_MIN = 0.45          # below this, the model is asked whether the passages answer it
_COVERAGE_WEIGHT = 6.0        # ranking: bonus for a chunk that contains every question term
_ROWHIT_WEIGHT = 1.5          # ranking: bonus for question terms landing on a table row
_COVERAGE_CERTAIN = 0.75      # at or above this, the lexical match is trusted without a model call

# ------------------------------------------------------------------ small text helpers


def _ws(text: Any) -> str:
    """Collapse whitespace and normalise the unicode the corpus uses (nbsp, curly quotes)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).replace("\xa0", " ")).strip()


_MONTHS = {m.lower() for m in ("January February March April May June July August September "
                               "October November December").split()}

# Words that never make up a company name on their own.
_NAME_STOP = _MONTHS | {
    # question words and function words
    "which", "what", "who", "whom", "whose", "how", "why", "when", "where", "in", "at", "on", "of",
    "the", "a", "an", "is", "was", "were", "are", "did", "does", "do", "had", "has", "have", "its",
    "their", "this", "that", "for", "from", "by", "with", "and", "or", "as", "to", "per", "than",
    "usd", "us", "fy", "q", "q1", "q2", "q3", "q4", "ceo", "cfo", "coo", "cto",
    # generic finance and reporting vocabulary
    "chief", "executive", "officer", "board", "audit", "committee", "call", "report", "reports",
    "annual", "quarter", "quarterly", "fourth", "first", "second", "third", "full", "year", "years",
    "results", "review", "rating", "target", "price", "forecast", "estimate", "estimates", "guidance",
    "outlook", "share", "shares", "repurchase", "programme", "program", "million", "millions",
    "billion", "billions", "percent", "value", "total", "net", "gross", "operating", "income",
    "revenue", "cash", "assets", "equivalents", "employees", "company", "city", "headquartered",
    "management", "analyst", "firm", "broker", "growth", "margin", "profit", "loss", "debt",
    "borrowing", "borrowings", "receivable", "receivables", "segment", "region", "dividend",
    "dividends", "capital", "securities", "equity", "research", "financial", "statements", "audited",
    "auditor", "case",
    # generic verbs and comparatives
    "much", "many", "end", "hold", "held", "reported", "spend", "spent", "announce", "announced",
    "guide", "guided", "set", "give", "gives", "earn", "earned", "generate", "generated", "pay",
    "paid", "previous", "new", "current", "higher", "lower", "more", "less", "compare", "compared",
    "both", "two", "during", "period",
}

_CONNECTORS = {"of", "and", "&", "the"}

# Last words of a broker or an audit firm ("BuyFin Capital", "Whitlock Byrne LLP"); a run ending in one
# of these is not the company under review. Kept generic on purpose: no name from the corpus.
_FIRM_TAILS = {"capital", "securities", "research", "partners", "bank", "trust", "llp", "advisors",
               "advisory", "investments", "markets", "asset", "management"}

# Geographies and reporting segments are capitalised like names but are not companies. Both sets are
# generic report vocabulary (continents, "segment", "division"), not data from this question set.
_REGION_WORDS = {"north", "south", "east", "west", "central", "america", "americas", "europe",
                 "africa", "asia", "pacific", "middle", "latin", "emea", "apac", "mena", "china",
                 "japan", "india", "korea", "singapore", "australia", "canada", "mexico", "brazil",
                 "germany", "france", "italy", "spain", "netherlands", "uk", "united", "states",
                 "kingdom", "and", "of", "the"}
_SEGMENT_NOUNS = {"segment", "division", "unit", "business", "arm", "subsidiary", "product",
                  "category", "line", "region", "market"}

# Words that can end a firm's name ("... Securities", "... Equity Research", "... Corp"): a run keeps
# consuming them so the whole firm name is seen and can be excluded as a broker or an audit firm.
_FIRM_SUFFIX_WORDS = _FIRM_TAILS | {"equity", "llp", "inc", "corp", "corporation", "ltd", "limited",
                                    "plc", "llc", "co", "sa", "nv"}


def _company_key(name: Any) -> str:
    """Company name without case, punctuation or legal suffix ('Northgate Robotics Inc.' -> 'northgate robotics')."""
    return " ".join(w for w in _ws(name).lower().replace("&", " and ").split()
                    if re.sub(r"[^a-z]", "", w) not in {"inc", "corp", "corporation", "ltd", "limited",
                                                        "plc", "llc", "co", "sa", "nv"})


def _first_word(key: str) -> str:
    parts = key.split()
    return parts[0] if parts else ""


# ------------------------------------------------------------------ table parsing


def _cell_text(cell) -> str:
    return _ws(cell.get_text(" ", strip=True))


def _cell_span(cell, name: str) -> int:
    try:
        return max(1, int(cell.get(name, 1) or 1))
    except (TypeError, ValueError):
        return 1


def _table_grid(table) -> list[list[str]]:
    """The table as a grid of strings, repeating a cell over the columns its colspan covers."""
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}  # column -> (text, rows still to fill)
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        row: list[str] = []
        col = 0

        def fill_pending() -> None:
            nonlocal col
            while col in pending:
                text, left = pending[col]
                row.append(text)
                if left <= 1:
                    del pending[col]
                else:
                    pending[col] = (text, left - 1)
                col += 1

        fill_pending()
        for cell in cells:
            fill_pending()
            text = _cell_text(cell)
            for _ in range(_cell_span(cell, "colspan")):
                row.append(text)
                if _cell_span(cell, "rowspan") > 1:
                    pending[col] = (text, _cell_span(cell, "rowspan") - 1)
                col += 1
        fill_pending()
        grid.append(row)
    return grid


def _column_headers(table, grid: list[list[str]], n_cols: int) -> list[str]:
    """The column headers, from <thead> when present, else the first row if it is all headers."""
    header_rows: list[list[str]] = []
    thead = table.find("thead")
    if thead is not None:
        for tr in thead.find_all("tr"):
            header_rows.append([_cell_text(c) for c in tr.find_all(["th", "td"], recursive=False)])
    elif grid:
        first_tr = table.find("tr")
        cells = first_tr.find_all(["th", "td"], recursive=False) if first_tr else []
        if cells and all(c.name == "th" for c in cells):
            header_rows.append([_cell_text(c) for c in cells])
    headers = [""] * n_cols
    for row in header_rows:
        for i in range(min(len(row), n_cols)):
            if row[i] and row[i] not in headers[i]:
                headers[i] = f"{headers[i]} {row[i]}".strip() if headers[i] else row[i]
    return headers


def _one_table_to_text(table) -> str:
    """Caption line first, then one line per table row with its label and every column header."""
    caption = ""
    cap = table.find("caption")
    if cap is not None:
        caption = _cell_text(cap)

    grid = _table_grid(table)
    n_cols = max((len(r) for r in grid), default=0)
    headers = _column_headers(table, grid, n_cols)

    # rows that are a single cell spanning the table are section titles ("Assets", "Current assets")
    body: list[list[str]] = []
    for tr, row in zip(table.find_all("tr"), grid):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) == 1 and _cell_span(cells[0], "colspan") >= 2 and row and row[0]:
            body.append(["__section__", row[0]])
        else:
            body.append(row)
    header_len = 0
    thead = table.find("thead")
    if thead is not None:
        header_len = len(thead.find_all("tr"))

    lines: list[str] = []
    section = ""
    for row in body[header_len:]:
        if not row:
            continue
        if row[0] == "__section__":
            section = row[1]
            lines.append(row[1])
            continue
        label = row[0]
        if not label:
            label = section
        if not label:
            continue
        pairs = [(headers[i] if i < len(headers) else "", value)
                 for i, value in enumerate(row[1:], start=1) if value]
        if not pairs:
            lines.append(label)
            continue
        joined = f"{label} | " + " | ".join(f"{h}: {v}" if h else v for h, v in pairs)
        if len(joined) <= MAX_LINE_CHARS:
            lines.append(joined)
        else:  # one line per value keeps every line short and free of other row labels
            lines.extend(f"{label} | {h}: {v}" if h else f"{label} | {v}" for h, v in pairs)

    out: list[str] = []
    if caption:
        if len(caption) <= MAX_LINE_CHARS:
            out.append(caption)
        else:  # wrap on spaces: squash() joins the lines back into the caption
            line = ""
            for word in caption.split():
                if len(line) + len(word) + 1 > MAX_LINE_CHARS:
                    out.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                out.append(line)
    out.extend(lines)
    return "\n".join(out)


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    if not table_html or not str(table_html).strip():
        return ""
    table = BeautifulSoup(str(table_html), "lxml").find("table")
    if table is None:
        return ""
    return _one_table_to_text(table)


# ------------------------------------------------------------------ corpus entities

_ENTITIES: dict[str, Any] = {"companies": {}, "brokers": set(), "auditors": set(),
                             "years": defaultdict(set), "docs": []}


def _register_company(name: str) -> str:
    key = _company_key(name)
    if key and key not in _ENTITIES["companies"]:
        _ENTITIES["companies"][key] = _ws(name)
    return key


def _strip_parenthetical(text: str) -> str:
    return _ws(re.sub(r"\s*\([^)]*\)", " ", _ws(text)))


def _detect_doc_type(soup, text: str) -> str:
    head = _ws(text[:400]).lower()
    title = _ws(soup.find("h1").get_text(" ", strip=True)).lower() if soup.find("h1") else ""
    blob = f"{title} {head}"
    if "results review" in blob or soup.find("table", class_="rr_summary") is not None:
        return "research_report"
    if "earnings call" in blob or "conference call" in blob or "earnings conference" in blob:
        return "earnings_call"
    return "annual_report"


def _detect_fiscal_year(doc_type: str, text: str) -> int:
    head = _ws(text[:600])
    patterns = [r"Annual Report\s+(\d{4})", r"Full Year\s+(\d{4})", r"FY\s?(\d{4})",
                r"(\d{4})\s+results review"]
    for pattern in patterns:
        match = re.search(pattern, head, re.I)
        if match:
            return int(match.group(1))
    match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", head)
    return int(match.group(1)) if match else 0


def _detect_broker(soup) -> str:
    first = soup.find("h1")
    for element in soup.find_all(string=True):
        if first is not None and element.parent is not None and first in element.parent.parents:
            break
        line = _ws(element)
        if line and "|" in line:
            return _ws(line.split("|")[0])
    return ""


def _detect_auditors(text: str) -> set[str]:
    names = set()
    for match in re.finditer(r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})\s+LLP\b", text):
        name = _ws(match.group(1))
        if name:
            names.add(name)
    return names


def _doc_header(meta: dict, section: str, caption: str = "") -> str:
    """A short header line: metadata plus the section (and table caption) of the chunk."""
    if not CONFIG["context_header"]:
        return ""
    label = section or caption
    kind = {"annual_report": "annual report", "earnings_call": "earnings call",
            "research_report": "research report"}.get(meta.get("doc_type", ""), "")
    parts = [str(meta.get("company", "")), f"FY{meta.get('fiscal_year')}", kind]
    if caption and caption != label:
        parts.append(caption)
    if label:
        parts.append(label)
    return " | ".join(p for p in parts if p)


# ------------------------------------------------------------------ chunking


def _split_long(text: str, limit: int) -> list[str]:
    """Split text on paragraph, then line, then word boundaries, staying under `limit` characters."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buffer = ""
    for paragraph in text.split("\n"):
        if len(buffer) + len(paragraph) + 1 <= limit:
            buffer = f"{buffer}\n{paragraph}" if buffer else paragraph
            continue
        if buffer:
            pieces.append(buffer)
        while len(paragraph) > limit:
            cut = paragraph.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        buffer = paragraph
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if p.strip()]


def _emit_table_chunks(table, meta: dict, section: str) -> list[Document]:
    caption = ""
    cap = table.find("caption")
    if cap is not None:
        caption = _cell_text(cap)
    if CONFIG["table_to_text"]:
        text = _one_table_to_text(table)
    else:  # B0: what get_text() gives, one cell per line, no row or column labels
        text = "\n".join(_cell_text(cell) for cell in table.find_all(["th", "td"]))
    header = _doc_header(meta, section, caption)
    docs: list[Document] = []
    buffer = header
    for line in text.split("\n"):
        if len(buffer) + len(line) + 1 > _TABLE_CHUNK_CHARS and buffer.strip():
            docs.append(Document(page_content=buffer.strip(), metadata=dict(meta)))
            buffer = header
        buffer = f"{buffer}\n{line}" if buffer else line
    if buffer.strip():
        docs.append(Document(page_content=buffer.strip(), metadata=dict(meta)))
    return docs


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["nav", "footer", "script", "style", "noscript"]):
        tag.decompose()

    body_text = soup.get_text("\n", strip=True)
    h1 = soup.find("h1")
    company = _strip_parenthetical(_cell_text(h1)) if h1 is not None else ""
    if not company:  # no headline: fall back to the first line
        company = _strip_parenthetical(body_text.split("\n")[0] if body_text else "")
    doc_type = _detect_doc_type(soup, body_text)
    fiscal_year = _detect_fiscal_year(doc_type, body_text)
    if doc_type == "research_report" and not fiscal_year:
        fiscal_year = 2025
    if doc_type == "earnings_call" and not fiscal_year:
        fiscal_year = 2025

    meta = {"source": file_name, "company": company, "fiscal_year": int(fiscal_year),
            "doc_type": doc_type}
    _register_company(company)
    if doc_type == "research_report":
        broker = _detect_broker(soup)
        if broker:
            _ENTITIES["brokers"].add(_company_key(broker))
    _ENTITIES["auditors"] |= {_company_key(a) for a in _detect_auditors(body_text)}
    _ENTITIES["docs"].append({"file": file_name, "company_key": _company_key(company),
                              "fiscal_year": int(fiscal_year), "doc_type": doc_type})

    # ---- B0/V1: flat chunking, get_text() + RecursiveCharacterTextSplitter, as in Tutorial 7
    if not CONFIG["structured_chunking"]:
        flat = BeautifulSoup(html, "lxml")
        for tag in flat(["script", "style", "noscript"]):
            tag.decompose()
        for table in flat.find_all("table"):
            if table.find_parent("table") is None:
                body = (_one_table_to_text(table) if CONFIG["table_to_text"]
                        else "\n".join(_cell_text(cell) for cell in table.find_all(["th", "td"])))
                table.replace_with(NavigableString(f"\n{body}\n"))
        text = flat.get_text("\n", strip=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=_NAIVE_CHUNK_CHARS,
                                                  chunk_overlap=_NAIVE_CHUNK_OVERLAP)
        return [Document(page_content=piece, metadata=dict(meta))
                for piece in splitter.split_text(text) if piece.strip()]

    # ---- walk the document in order, keeping tables apart from the narrative
    chunks: list[Document] = []
    section = ""
    buffer_parts: list[str] = []
    buffer_len = 0

    def flush_narrative() -> None:
        nonlocal buffer_parts, buffer_len
        if not buffer_parts:
            return
        text = "\n".join(buffer_parts).strip()
        buffer_parts, buffer_len = [], 0
        if not text:
            return
        header = _doc_header(meta, section)
        budget = _MAX_BLOCK_CHARS - len(header) - 1
        for piece in _split_long(text, max(200, budget)):
            content = f"{header}\n{piece}" if header else piece
            chunks.append(Document(page_content=content[:MAX_CHUNK_CHARS], metadata=dict(meta)))

    if soup.body is not None:
        elements = soup.body.find_all(["h1", "h2", "h3", "p", "li", "table"])
    else:
        elements = soup.find_all(["h1", "h2", "h3", "p", "li", "table"])

    for element in elements:
        if element.name == "table":
            if element.find_parent("table") is not None:
                continue
            flush_narrative()
            chunks.extend(_emit_table_chunks(element, meta, section))
            continue
        if element.find_parent(["p", "li", "table"]) is not None:
            continue  # nested element already covered by its parent
        if element.name in ("h1", "h2", "h3"):
            text = _cell_text(element)
            if text:
                flush_narrative()
                section = text
            continue
        text = _cell_text(element)
        if not text:
            continue
        if buffer_len + len(text) + 1 > _MAX_BLOCK_CHARS:
            flush_narrative()
        buffer_parts.append(text)
        buffer_len += len(text) + 1
    flush_narrative()

    # safety net: nothing above may exceed the hard limit
    final: list[Document] = []
    for chunk in chunks:
        if len(chunk.page_content) <= MAX_CHUNK_CHARS:
            final.append(chunk)
        else:
            for piece in _split_long(chunk.page_content, MAX_CHUNK_CHARS - 10):
                final.append(Document(page_content=piece, metadata=dict(chunk.metadata)))
    return final


# ------------------------------------------------------------------ query parsing

_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)")
_RESEARCH_RE = re.compile(
    r"\b(target price|price target|rating|results review|broker|broker's|forecast|forecasts|"
    r"estimate|estimates|upside|analyst|analysts|equity research|investment case)\b")
_EARNINGS_RE = re.compile(
    r"\b(earnings call|conference call|earnings conference|fourth[- ]quarter|q4 call|call transcript|"
    r"earnings call transcript)\b")


def _name_runs(question: str) -> list[dict]:
    """Runs of capitalised words, each with the words on either side and how it ended.

    A possessive ends a run ("Brightwater Energy's Midstream segment" is one company plus a segment
    name), so the segment never gets glued onto the company. The neighbours are what tell a company
    from a geography ("... in Asia Pacific") or a reporting segment ("... Bakery segment").
    """
    text = _ws(question).replace("\u2019", "'")
    text = re.sub(r"'s\b", " | ", text)
    text = re.sub(r"'(?=\s|$)", " | ", text)
    tokens = [raw.strip("()[]{}<>\"'.,;:!?") for raw in text.split()]

    runs: list[dict] = []
    current: list[str] = []
    start = 0

    def close(next_word: str, possessive: bool) -> None:
        nonlocal current
        if not current:
            return
        runs.append({"text": " ".join(current),
                     "prev": tokens[start - 1] if start else "",
                     "prev2": tokens[start - 2] if start > 1 else "",
                     "next": next_word, "at_start": start == 0, "possessive": possessive})
        current = []

    for i, token in enumerate(tokens):
        if not token:
            close("", False)
            continue
        if token == "|":
            close("", True)  # the run was cut by a possessive
            continue
        if current and token.lower() in _CONNECTORS:
            current.append(token)
            continue
        if current and token.lower() in _FIRM_SUFFIX_WORDS and re.fullmatch(r"[A-Za-z&.\-]+", token):
            current.append(token)  # "... Securities", "... Equity Research": keep the firm name whole
            continue
        if re.fullmatch(r"[A-Z][A-Za-z&.\-]*", token) and token.lower() not in _NAME_STOP:
            if not current:
                start = i
            current.append(token)
            continue
        close(token.lower(), False)
    close("", False)
    return runs


def _parse_companies(question: str) -> list[str]:
    """Task 2.2: every company the question names, whether or not it is in the corpus.

    The default is to keep a capitalised name run; what is dropped is what surrounds the company
    under review: a broker or an audit firm (from the entities the corpus itself provided), a
    geography, a reporting segment, or a bare capitalised word that opens the question without
    owning anything. Nothing here depends on how a particular question is worded.
    """
    known: dict[str, str] = _ENTITIES["companies"]
    firsts = {_first_word(key) for key in known}
    brokers = _ENTITIES["brokers"] | _ENTITIES["auditors"]
    companies: list[str] = []
    seen: set[str] = set()

    for run in _name_runs(question):
        name = run["text"]
        key = _company_key(name)
        if not key or key in seen:
            continue
        words = key.split()
        if key in brokers or any(key.endswith(" " + broker) for broker in brokers):
            continue  # a broker or an audit firm named in the question
        if words[-1] in _FIRM_TAILS:
            continue  # a broker or an audit firm rather than the company under review
        if all(word in _REGION_WORDS for word in words):
            continue  # a geography ("Asia Pacific"), not a company
        if run["next"] in _SEGMENT_NOUNS:
            continue  # "Midstream segment", "Bakery division"
        if run["prev"] in {"at", "to"} and run["prev2"] in {"analyst", "according"}:
            continue  # "the analyst at ...", "according to ..."
        if len(words) == 1 and key not in known and key not in firsts:
            # a single word is only trusted when it owns something ("Evergreen's revenue") or sits
            # inside the sentence; a bare capitalised word that opens the question usually is not one
            if run["at_start"] and not run["possessive"]:
                continue
        seen.add(key)
        companies.append(name)
    return companies


def _llm_companies(question: str) -> list[str]:
    """Fallback only: ask the model for the company names when the rules found none."""
    try:
        llm = make_llm(max_tokens=80)
        prompt = ("List the companies the question below asks about, separated by commas, using the "
                  "names exactly as written. Do not list brokers, audit firms or analysts. If there "
                  "are none, reply NONE.\nQuestion: " + question + "\nCompanies:")
        answer = _ws(llm.invoke(prompt).content)
    except Exception:
        return []
    if not answer or answer.strip().upper() in {"NONE", "NOT_FOUND"}:
        return []
    names = [_ws(x) for x in re.split(r"[,;\n]| and ", answer) if _ws(x)]
    return names[:4]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    text = _ws(question)
    lowered = text.lower()

    companies = _parse_companies(text)
    if not companies and _ENTITIES["companies"] and len(_name_runs(text)) > 0:
        companies = _llm_companies(text)

    match = _YEAR_RE.search(text)
    fiscal_year = int(match.group(1)) if match else None

    if _EARNINGS_RE.search(lowered):
        doc_type = "earnings_call"
    elif _RESEARCH_RE.search(lowered):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    return {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}


# ------------------------------------------------------------------ index

_STOPWORDS = set("""a an the and or of to in on at for from by with as is are was were be been being
this that these those it its their they we our you your i he she his her them us what which who whom
whose how much many did does do done has have had having can could would should may might must will
shall about above after again against all am any because before below between both but into more most
no nor not only other out over same so some such than then there through under until up very when
where while why also just company companies firm firms group inc corp ltd plc llc co usd million
millions billion billions percent percentage per share shares fy q1 q2 q3 q4 quarter quarterly
annual report reports year years full end during period""".split())


def _tok(text: str) -> list[str]:
    """Lower-case word and number tokens; numbers keep decimals, plurals fold to the singular."""
    if not CONFIG["better_tokenizer"]:
        return str(text).lower().split()
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"'(?=\s|$)", "", text)
    text = text.replace("'", "")
    out: list[str] = []
    for word in re.findall(r"[a-z][a-z'\-]*|\d+(?:[.,]\d+)*", text):
        word = word.strip("'-")
        if not word or (word in _STOPWORDS and not word[0].isdigit()):
            continue
        if word[0].isdigit():
            word = word.replace(",", "")
            out.append(word)
            if "." in word and word.split(".")[0] not in _STOPWORDS:
                out.append(word.split(".")[0])
            continue
        out.append(word)
        if len(word) > 4 and word.endswith("ies"):
            out.append(word[:-3] + "y")
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            out.append(word[:-1])
    return out


def _chunk_body(chunk: Document) -> str:
    """The chunk text as it is: the header is part of what the corpus says about the chunk.

    Kept as a hook so the ablation can switch between "index the header" and "index only the body".
    """
    return chunk.page_content


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    from rank_bm25 import BM25Okapi

    token_lists = [_tok(_chunk_body(c)) for c in chunks]
    bm25 = BM25Okapi(token_lists, k1=1.5, b=0.75) if token_lists else None

    df: dict[str, int] = defaultdict(int)
    for tokens in token_lists:
        for token in set(tokens):
            df[token] += 1
    n_docs = max(len(token_lists), 1)
    idf = {word: math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0) for word, count in df.items()}
    idf_default = math.log((n_docs + 0.5) / 0.5 + 1.0)  # a word that never occurs in the corpus

    by_company: dict[str, list[int]] = defaultdict(list)
    by_company_type: dict[tuple[str, str], list[int]] = defaultdict(list)
    years: dict[tuple[str, str], set[int]] = defaultdict(set)
    tokensets = [set(t) for t in token_lists]
    for i, chunk in enumerate(chunks):
        key = _company_key(chunk.metadata.get("company", ""))
        doc_type = chunk.metadata.get("doc_type", "")
        by_company[key].append(i)
        by_company_type[(key, doc_type)].append(i)

    # which years a company's documents of each type actually state: an annual report shows its own
    # fiscal year and the prior year, an earnings call its own year and the FY+1 guidance, a research
    # report its own year and the two forecast years.
    for record in _ENTITIES["docs"]:
        key, doc_type, fiscal_year = record["company_key"], record["doc_type"], record["fiscal_year"]
        pairs = {fiscal_year}
        if doc_type == "annual_report":
            pairs.add(fiscal_year - 1)
        elif doc_type == "earnings_call":
            pairs.add(fiscal_year + 1)
        elif doc_type == "research_report":
            pairs.update({fiscal_year + 1, fiscal_year + 2})
        years[(key, doc_type)] |= pairs

    return {"chunks": chunks, "bm25": bm25, "df": dict(df), "idf": idf, "idf_default": idf_default,
            "tokensets": tokensets, "by_company": dict(by_company),
            "by_company_type": dict(by_company_type), "years": dict(years),
            "companies": dict(_ENTITIES["companies"])}


# ------------------------------------------------------------------ retrieval

_ABSTENTION_EXTRA = {"usd", "million", "millions", "billion", "billions", "percent", "percentage",
                     "share", "shares", "fy", "q", "q4", "full", "year", "annual", "report", "results",
                     "review", "company", "much", "many", "did", "does", "what", "which", "who",
                     "large", "small", "big", "announce", "announced"}

# The share of a question's content words (IDF-weighted) that the best passages must contain. Below
# it the corpus does not discuss what is asked, however the question is worded; above it the model
# check below decides. A general threshold, not a list of phrases from any question set.
_COVERAGE_QUESTION_MIN = 0.30


# Words that only say which document to read (handled by doc_type filtering), not what to find in it.
_DOC_WORDS = {"earnings", "earning", "call", "calls", "conference", "fourth-quarter", "fourth", "quarter",
              "management", "transcript", "broker", "results", "review", "set", "give"}

# Question wording -> the wording the reports use (query expansion for BM25 and coverage).
_SYNONYMS = {
    "guide": ["outlook", "expect", "guidance"], "guided": ["outlook", "expect", "guidance"],
    "guidance": ["outlook", "expect"], "headquartered": ["headquarters"], "city": ["headquarters"],
    "headcount": ["employees"], "staff": ["employees"], "held": ["hold"], "workforce": ["employees"],
    "buyback": ["repurchase"], "auditor": ["audited"], "audit": ["audited"],
}


def _expand(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for token in tokens:
        out.extend(_SYNONYMS.get(token, []))
    return out


def _content_words(question: str, parsed: dict) -> list[str]:
    drop: set[str] = set(_DOC_WORDS)
    for name in parsed.get("companies", []):
        drop |= set(_tok(name))
    if parsed.get("fiscal_year"):
        drop.add(str(parsed["fiscal_year"]))
    words = [w for w in _tok(question) if w not in drop and w not in _ABSTENTION_EXTRA]
    words = [w for w in words if not w.isdigit()]
    if CONFIG["better_tokenizer"]:
        # a synonym the corpus uses replaces a question word it never uses
        expanded = []
        for w in words:
            alternatives = _SYNONYMS.get(w)
            expanded.extend(alternatives[:1] if alternatives else [w])
        words = expanded
    return words


def _coverage(index: dict, chunk_idx: int, content: list[str]) -> float:
    if not content:
        return 1.0
    total = sum(index["idf"].get(w, index["idf_default"]) for w in content)
    if total <= 0:
        return 1.0
    tokens = index["tokensets"][chunk_idx]
    hit = sum(index["idf"].get(w, index["idf_default"]) for w in content if w in tokens)
    return hit / total


def _answerable(question: str, chunks: list[Document]) -> bool:
    """Ask the model whether the retrieved passages actually state the answer (Task 3, soft layer)."""
    if not chunks:
        return False
    context = "\n\n".join(f"[{i}] {format_context([c])[4:]}" for i, c in enumerate(chunks, 1))
    prompt = ("A question and some passages from company reports are below. Decide only whether the "
              "passages state the fact the question asks for. Reply YES if one passage answers it "
              "directly (the figure or the name itself is printed there), NO otherwise (the passage "
              "is only about the same topic, or has the number for another year or another company).\n\n"
              f"Question: {question}\n\nPassages:\n{context}\n\nAnswer (YES or NO):")
    try:
        reply = _ws(make_llm(max_tokens=4).invoke(prompt).content).upper()
    except Exception:
        return True  # never abstain because of an API hiccup
    return reply.startswith("Y")


def _row_hits(chunk: Document, content: list[str]) -> int:
    """How many question terms appear on a table row of the chunk (a line that carries a `label | value`)."""
    lines = [line.lower() for line in chunk.page_content.split("\n")[1:] if "|" in line]
    return sum(1 for word in content if any(word in line for line in lines))


def _score(index: dict, chunk_idx: int, question_tokens: list[str], year: int | None,
           bm25_scores, content: list[str] | None = None) -> float:
    chunk = index["chunks"][chunk_idx]
    score = float(bm25_scores[chunk_idx])
    if not CONFIG["hybrid_scoring"]:
        return score  # B0/V1/V2/V3: plain BM25
    if year is not None:
        if f"fy{year}" in chunk.page_content.lower() or re.search(rf"\b{year}\b", chunk.page_content):
            score += 1.0
        if chunk.metadata.get("fiscal_year") == year:
            score += 1.5
        elif chunk.metadata.get("fiscal_year") == (year or 0) + 1 and chunk.metadata.get("doc_type") == "annual_report":
            score += 0.4
    if content:
        # a chunk that contains every term the question asks about is a far better candidate than one
        # that merely repeats a common word many times
        score += _COVERAGE_WEIGHT * _coverage(index, chunk_idx, content)
        score += _ROWHIT_WEIGHT * _row_hits(chunk, content)
    return score


def _rank_pool(index: dict, pool: list[int], question_tokens: list[str], year: int | None,
               bm25_scores, content: list[str] | None = None) -> list[int]:
    def score_of(i: int) -> float:
        return _score(index, i, question_tokens, year, bm25_scores, content)

    ranked = sorted(pool, key=score_of, reverse=True)
    if not ranked:
        return []
    if not CONFIG["compact_context"]:
        return ranked
    best = score_of(ranked[0])
    return [i for i in ranked if score_of(i) >= best * _SCORE_KEEP]


def _dedupe(ranked: list[int], index: dict) -> list[int]:
    """Keep one chunk per (company, table caption / section) so a table is not handed over twice."""
    if not CONFIG["compact_context"]:
        return list(ranked)
    seen: set[tuple] = set()
    out: list[int] = []
    for i in ranked:
        chunk = index["chunks"][i]
        caption = ""
        match = re.search(r"\| ([^|\n]+)$", chunk.page_content.split("\n")[0])
        if match:
            caption = match.group(1).strip().lower()
        key = (_company_key(chunk.metadata.get("company", "")), caption)
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    if not index or not index.get("chunks"):
        return []
    if not any(CONFIG.get(k, True) for k in ("metadata_filter", "hybrid_scoring", "context_header",
                                             "better_tokenizer", "per_company_retrieval",
                                             "abstention", "compact_context")):
        return _naive_retrieve(index, question)  # B0: plain BM25 over the whole corpus, never []
    parsed = parse_query(question)
    if not CONFIG["metadata_filter"]:  # B0/V1/V2 do not parse the question at all
        parsed = {"companies": [], "fiscal_year": None, "doc_type": "annual_report"}
    year = parsed.get("fiscal_year")
    doc_type = parsed.get("doc_type", "annual_report")
    named = parsed.get("companies", [])
    known_companies = index["companies"]

    known: list[str] = []
    for name in named:
        key = _company_key(name)
        if key in known_companies:
            known.append(key)
        elif _first_word(key) in {_first_word(k) for k in known_companies}:
            # looks like a corpus company but is not in the corpus: abstain
            if CONFIG["abstention"]:
                return []
        elif named == [name]:
            if CONFIG["abstention"]:
                return []
        else:
            if CONFIG["abstention"]:
                return []

    if CONFIG["abstention"] and known and year is not None:
        covered: set[int] = set()
        for key in known:
            covered |= index["years"].get((key, doc_type), set())
            covered |= index["years"].get((key, "annual_report"), set())
        if covered and year not in covered:
            return []

    question_tokens = _expand(_tok(question)) if CONFIG["better_tokenizer"] else _tok(question)
    bm25_scores = index["bm25"].get_scores(question_tokens) if index["bm25"] is not None else None
    content = _content_words(question, parsed)
    compact = CONFIG["compact_context"]
    max_return = _MAX_FINAL if compact else MAX_RETRIEVED

    def pool_for(key: str) -> list[int]:
        if not CONFIG["metadata_filter"]:
            return list(range(len(index["chunks"])))
        pool = index["by_company_type"].get((key, doc_type), [])
        if len(pool) < 2:  # fall back to the company's other document types
            pool = index["by_company"].get(key, [])
        return pool

    comparison = len(known) > 1 and CONFIG["per_company_retrieval"]
    if not known:
        ranked = _rank_pool(index, list(range(len(index["chunks"]))), question_tokens, year, bm25_scores,
                            content)
        picked = _dedupe(ranked, index)[:max_return]
    elif comparison:
        picked = []
        per_company = max(1, max_return // len(known))
        for key in known:
            ranked = _dedupe(_rank_pool(index, pool_for(key), question_tokens, year, bm25_scores, content),
                             index)
            picked.extend(ranked[:min(per_company, _PREFERRED_ROWS_CMP)])
        picked = picked[:max_return]
    else:
        ranked = []
        for key in known:
            ranked.extend(_rank_pool(index, pool_for(key), question_tokens, year, bm25_scores, content))
        ranked.sort(key=lambda i: _score(index, i, question_tokens, year, bm25_scores, content), reverse=True)
        cap = min(_PREFERRED_ROWS, max_return) if compact else max_return
        picked = _dedupe(ranked, index)[:cap]

    if not picked:
        return []

    if CONFIG["abstention"]:
        best = max(_coverage(index, i, content) for i in picked[:3]) if content else 1.0
        if best < _COVERAGE_QUESTION_MIN:
            return []  # the question's own vocabulary is almost absent: nothing here answers it
        if best < _COVERAGE_CERTAIN and not _answerable(question, [index["chunks"][i] for i in picked[:3]]):
            return []

    return [index["chunks"][i] for i in picked[:max_return]]


def _naive_retrieve(index: dict, question: str) -> list[Document]:
    """B0 baseline only: BM25 top-5 over the whole corpus, no parsing, no filter, never []."""
    if index["bm25"] is None:
        return []
    scores = index["bm25"].get_scores(_tok(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:MAX_RETRIEVED]
    return [index["chunks"][i] for i in order]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
