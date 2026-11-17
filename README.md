# FTEC5660 Homework 4 — solution

Retrieval for a RAG assistant over company reports: HTML parsing, chunking with metadata, query
parsing, lexical retrieval with a hard cap of five chunks, and abstention.

`python3 hw4.py --data-folder public_test` runs the whole pipeline and writes `results.csv`.
`python3 ablation.py --data-folder public_test` reproduces the table in the ablation section and
writes `ablation_results.csv`.

**Public test set (60 questions, 40 documents): 75.0 / 75** on the automated tasks; the run takes
about 22 s, of which 16 s is the model-based answerability check and the generator.

---

## 1. Pipeline

```
corpus/doc_038.html
        │
        ▼
 parse (BeautifulSoup + lxml, nav/footer/script removed)
        │   h1 → company, headline → doc_type, headline → fiscal_year
        │   the body is walked in order: headings, paragraphs, lists and <table>
        │
        ├── narrative blocks ──▶ grouped up to 1,200 chars, split on paragraph/word boundaries
        │
        └── each <table> ──▶ table_to_text(): caption line, then one line per table row,
                             "Revenue | 2025: 2,710.4 | 2024: 2,629.1"
                             (colspan/rowspan expanded, section rows kept as titles)
        │
        ▼
 chunk_document: one chunk per table, one chunk per narrative block, each ≤ 1,500 chars,
 with metadata source / company / fiscal_year / doc_type, and a header line
 "Northgate Robotics | FY2025 | annual report | 6. Financial statements — Consolidated balance sheet"
        │
        ▼
 build_index: BM25Okapi (k1=1.5, b=0.75) over the chunk text plus a per-company/per-year directory,
 DF and IDF tables, and the year coverage each company's document types actually state
        │
        ▼
 parse_query (rules, model only as a fallback) → companies, fiscal_year, doc_type
        │
        ▼
 retrieve
   1. hard abstention: a company that is not in the corpus, or a year the company's documents of
      that type do not state → []
   2. candidate pool: metadata filter on (company, doc_type); the company's other document types
      are a fallback when the pool is tiny
   3. score = BM25 + year bonus + 6·(IDF-weighted share of question terms present)
                    + 1.5·(question terms that land on a table row)
   4. comparison: one ranking per company, so each company keeps a chunk
   5. dedupe on (company, table caption/section); keep the top 3 (2 per company for a comparison),
      at most 4
   6. soft abstention: a phrasing that reports never disclose (lobbying, churn, …) → []; otherwise,
      unless the lexical coverage is clearly high, one deepseek-v4-flash YES/NO call asking whether
      the passages state the fact
        │
        ▼
 generate (fixed, temperature 0) → one number, one name, or NOT FOUND
```

**Why this shape.** The corpus is small and the questions are short factual lookups, so BM25 is
enough to find the passage once the candidate pool is right; what BM25 cannot do is tell
`Northgate Robotics` from `Northgate Payments` (both tokens contain "Northgate") or 2024 from 2025
(every table prints both years). That is exactly what the chunk metadata is for, so filtering
happens before scoring. The two additions to plain BM25 both attack the same failure: BM25 rewards a
chunk that repeats one common word many times, while a table row that contains the whole question —
"gross profit", "inventories" — is what the generator needs, so the score adds the IDF-weighted share
of question terms the chunk contains and a bonus for terms that appear on a table row. The pipeline
figure above is the one to read together with the table below: every step after B0 is one of the
rows.

---

## 2. Ablation study

Every row adds exactly one component to the row above it; `ablation.py` runs the real
`chunk_document` / `build_index` / `parse_query` / `retrieve` / `generate` under each configuration,
so the numbers come from the same code that is submitted. `recall` is Recall@5 on the 46 answerable
questions, `kept` and `rejected` are Task 3's shares, `acc` is end-to-end answer accuracy on all 60.

| # | Configuration | Change vs. previous row | Recall@5 | Kept | Rejected | Acc |
|---|---|---|---|---|---|---|
| B0 | naive RAG | `get_text()`, `RecursiveCharacterTextSplitter`, BM25 over the whole corpus, no parsing, never abstain | 15.2% | 100% | 35.7% | 35.0% |
| V1 | structured tables | `table_to_text()` instead of `get_text()` | 23.9% | 100% | 35.7% | 41.7% |
| V2 | metadata filtering | `parse_query()` + filter on company, year, document type | 84.8% | 100% | 35.7% | 86.7% |
| V3 | context header | metadata + section + table caption on every chunk | 97.8% | 100% | 35.7% | 96.7% |
| V4 | better tokenizer | numbers/years/entities, plural folding, synonym expansion | 95.7% | 100% | 35.7% | 96.7% |
| V5 | per-company retrieval | one ranking per company for comparisons | 100% | 100% | 35.7% | 100% |
| V6 | abstention | hard rules + coverage threshold + model YES/NO check | 100% | 100% | 100% | 100% |
| V7 | fewer chunks | at most 4 chunks, drop the tail below 35% of the best score | 100% | 100% | 100% | 100% |

Points by task for the final row: 2.2 = 10.0, 2.3 = 15.0, 3 = 10.0, 4 = 20.0 (Task 1.1 and 1.2 are
not affected by these switches; the submitted code scores 8.0 and 7.0).

### What each change did

- **V1, structured tables (+8.7 recall points).** Naive `get_text()` gives a table as a stream of
  numbers with no row or column attached, so even when the right table is retrieved the exact figure
  sits alone in a chunk and the generator has to guess the column. One line per row with its label
  and both column headers fixes both halves: the surrounding text now contains the row label so BM25
  can find it, and the generator can see which year a number belongs to.
- **V2, metadata filtering (+60.9 recall points, the largest single step).** This is the step that
  makes the task tractable. Without it, a query for "revenue FY2025" pulls chunks from every company
  and both annual reports of the right one; the five returned chunks are usually five copies of the
  same table. Filtering on the parsed company and document type takes the pool from 886 chunks to
  roughly 70, and the right table is almost always in the top few.
- **V3, context header (+13.0 recall points).** A table caption such as "Consolidated balance
  sheet at 31 December" is the words a question uses for that section ("total assets at 31 December
  2024"), but it is only printed once, above the table. Repeating it on every chunk of the table lets
  BM25 match the chunk for the right reason. The year in the header also makes the chunk's own fiscal
  year visible in its text.
- **V4, better tokenizer (−2.1 recall points: no gain here).** Keeping decimals and years as single
  tokens and folding plurals were meant to help, but on this corpus they do not: the plain
  lower-case whitespace tokeniser already separates the number tokens, and the corpus uses one
  consistent wording per fact. The small loss comes from stemming: "assets" and "asset" fold together
  but so do words the query needs to keep apart. Its one real contribution is the synonym map
  ("guide" → "outlook/expect", "headquartered" → "headquarters", "headcount" → "employees"), which
  the coverage component relies on; without it a question phrased with a word the report never uses
  scores zero coverage and is wrongly rejected. Kept for that reason, not for the recall number.
- **V5, per-company retrieval (+2.1 recall points, and it is why comparisons are at 100%).** A
  comparison needs two facts in two documents; a single ranking over a mixed pool reliably returns
  five chunks from the company whose name happens to score higher. Ranking each company separately
  and taking the top one or two from each guarantees both companies are represented.
- **V6, abstention (+64.3 rejected points, 3.6 → 10.0 points on Task 3).** Two hard rules cover 10
  of the 14 unanswerable questions with no effect on answerable ones: a company that is not in the
  corpus, and a year outside what a company's documents of that type state. The remaining four are
  undisclosed facts; a phrase list plus a model YES/NO check on the top three chunks covers them.
  The model check only runs when lexical coverage is not clearly high, which keeps it to about 19
  calls per run.
- **V7, fewer chunks (no change).** On the public set, four chunks and five chunks both give 100%,
  because the correct chunk is almost always first. It is kept because a shorter context is strictly
  less likely to mislead the generator when a look-alike passage is retrieved, and it is the setting
  used in `results.csv`, so the table matches the code.

### Error analysis

The remaining failures during development, and what fixed them, are the most instructive part:

1. **The year regex missed every `FY2025`.** `\b(20\d{2})\b` does not match inside "FY2025" because
   `Y` and `2` are both word characters, so 21 of 60 questions parsed `fiscal_year: null`, their
   year filter was skipped, and half of the answerable questions came back empty. The pattern is now
   `(?<!\d)(20\d{2})(?!\d)`. This single line took the run from 44 to 66 points.
2. **`"rating"` matched inside `"operating"`.** The document-type hints were substrings, so every
   question about *operating* income was classified as a research-report question and its pool
   became the broker note. Word-boundary regexes fixed it; Task 2.2 went to 100%.
3. **Possessives broke query expansion.** The tokenizer produced `software's` and `software'`, which
   appear nowhere in the corpus, and the IDF-weighted coverage of an otherwise perfect chunk fell to
   8%, so the answerability check rejected it. Stripping the possessive before tokenising took the
   score from 66 to 72.
4. **The right table was ranked 10th for "inventories".** BM25 scored the cover page highest because
   it repeats the company name. Adding the coverage term (share of question terms present) and the
   table-row bonus moved the balance sheet to first place: 22 of the 46 answerable questions became
   hits without any model call.
5. **The generator, not the retriever, was the limit in the last few cases.** With the same chunks,
   the remaining wrong answers were the generator reading the wrong column; giving it one chunk per
   table (dedupe) and at most four chunks in total removed them.