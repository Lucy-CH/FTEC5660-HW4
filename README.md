# FTEC5660 Homework 4 — solution

Retrieval for a RAG assistant over company reports: HTML parsing, chunking with metadata, query
parsing, lexical retrieval with a hard cap of five chunks, and abstention.

`python3 hw4.py --data-folder public_test` runs the whole pipeline and writes `results.csv`.
`python3 ablation.py --data-folder public_test` reproduces the table in the ablation section and
writes `ablation_results.csv`.

Every design choice in Task 5 is a switch in `hw4.CONFIG`; the submitted code runs with all of them
on, and `ablation.py` flips them one at a time, in the order of the table below, starting from a
configuration with every switch off (B0).

Task 6 (reflection) is in [Reflection.pdf](Reflection.pdf).

**Public test set (60 questions, 40 documents): 75.0 / 75** on the automated tasks; the run takes
about 18 s, of which 13 s is retrieval (19 answerability calls) and the rest the generator.

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

Every numbered step above is one switch in `hw4.CONFIG` (`table_to_text`, `structured_chunking`,
`metadata_filter`, `hybrid_scoring`, `context_header`, `better_tokenizer`,
`per_company_retrieval`, `abstention`, `compact_context`). The submitted code has them all on;
`ablation.py` turns them on one at a time, in the table's order, from an all-off configuration.

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

Every row turns on exactly one switch in `hw4.CONFIG` relative to the row above it, starting from a
configuration in which every switch is off. `ablation.py` runs the real `chunk_document` /
`build_index` / `parse_query` / `retrieve` / `generate` under each configuration, so the numbers come
from the code that is submitted, and the last row is the configuration `hw4.py` runs with. `recall`
is Recall@5 on the 46 answerable questions, `kept` and `rejected` are Task 3's shares, `acc` is
end-to-end answer accuracy on all 60, `chunks` is the mean number of passages handed to the
generator.

| # | Configuration | Change vs. previous row | Recall@5 | Kept | Rejected | Acc | chunks |
|---|---|---|---|---|---|---|---|
| B0 | naive RAG | `get_text()` (a table becomes a stream of cells), `RecursiveCharacterTextSplitter(1000, 150)`, `rank_bm25` top-5 over the whole corpus, no query parsing, never `[]` | 28.3% | 100% | 0% | 43.3% | 5.00 |
| V1 | `table_to_text` | each table becomes one line per row with its label and column headers; still flat-split | 23.9% | 100% | 0% | 41.7% | 5.00 |
| V2 | structured chunking | one chunk per table (split on row boundaries), narrative grouped by section instead of by character count | 21.7% | 100% | 0% | 38.3% | 5.00 |
| V3 | metadata filtering | `parse_query()`; the BM25 pool becomes the question's company and document type | 69.6% | 100% | 0% | 76.7% | 5.00 |
| V4 | hybrid scoring | BM25 + a year bonus + the IDF-weighted share of question terms present + a table-row bonus | 100.0% | 100% | 0% | 98.3% | 5.00 |
| V5 | context header | every chunk starts with company \ | FY \ | doc type \ | caption \ | section | 97.8% | 100% | 0% | 93.3% | 5.00 |
| V6 | better tokenizer | stop words, possessives, numbers without commas, plural folding, synonyms | 97.8% | 100% | 0% | 95.0% | 5.00 |
| V7 | per-company retrieval | a comparison runs one ranking per company | 100.0% | 100% | 0% | 96.7% | 4.80 |
| V8 | abstention | `[]` for an unknown company, an unstated year, an undisclosed-fact phrase, or a failed model YES/NO check | 100.0% | 100% | 100% | 98.3% | 3.63 |
| V9 | compact context | dedupe by table/section, 3 chunks (2 per company for a comparison), drop the tail below 35% of the best score | 100.0% | 100% | 100% | 100.0% | 2.47 |

Points by task for the final row: 2.2 = 10.0, 2.3 = 15.0, 3 = 10.0, 4 = 20.0 — 75.0 / 75 together
with Tasks 1.1 and 1.2, which these switches do not affect (`hw4.py` reports 8.00 and 7.00).

### What each change did

- **V1, structured tables (−4.4 recall points, −1.6 accuracy points at this row).** On its own, under
  plain BM25 over the whole corpus, writing each table as "row label | column: value" lines does not
  help: 35 misses against B0's 33. The row text is longer (343 → 402 chunks), and each row now
  repeats column headers such as "2025" and "2024" that every report shares, which probably adds
  noise rather than signal for BM25 (I did not isolate this further). The value of the change appears later: once the right table is retrieved (V3–V4), the
  generator needs the row label and the year next to each number to pick the right figure, which a
  stream of bare cells does not give it. It is also what Task 1.1 grades.
- **V2, structured chunking (−2.2 recall points, −3.4 accuracy points here).** One chunk per table
  and section-grouped narrative instead of 1,000-character windows grows the corpus from 402 to 865
  chunks, so a fixed top-5 over the whole corpus covers a smaller share of it, and recall dips again.
  Like V1, it is a prerequisite rather than a gain on its own: metadata filtering (V3), the per-table
  header (V5) and dedupe by table (V9) all work on whole tables and sections.
- **V3, metadata filtering (+47.9 recall points, the largest single step).** This is what makes the
  task tractable. Without it a query for "revenue FY2025" pulls chunks from every company and from
  both annual reports of the right one, and the five returned chunks are usually five look-alikes.
  Filtering the pool on the parsed company and document type takes it from 865 chunks to roughly
  70 annual-report chunks, or under 10 research-report or earnings-call chunks, and the answerable
  misses drop from 36 to 14 — all of them table questions (11) and comparisons (3).
- **V4, hybrid scoring (+30.4 recall points; all 14 misses recovered).** Plain BM25 over a filtered
  pool still ranks a cover page or a narrative section above a balance sheet, because those repeat
  the company name while the line item ("inventories") appears once, in the statements. The
  coverage term (the IDF-weighted share of the question's terms that the chunk actually contains)
  plus the table-row bonus move the right table to the top; the year bonus selects the column the
  question asks about. With both the pool and the scoring right, recall reaches 100% here.
- **V5, context header (−2.2 recall points, −5.0 accuracy points).** Repeating "company \| FY2025 \|
  annual report \| Consolidated balance sheet" on every chunk adds the section vocabulary a question
  uses, but it also repeats the company name and fiscal year on *every* chunk of a report, which
  makes chunks of the same report look more alike than before. Two questions regressed: a fact
  question ("In which city is Halcyon Freight headquartered?", q027) is no longer retrieved, because
  its answer sits in a short narrative paragraph that now competes with header-carrying chunks of the
  same report; and an unanswerable question ("Orchid Bay Resorts' operating income", q044 —
  abstention is still off at this row) gets passages whose headers look close enough to the question
  that the generator answers with a number instead of NOT_FOUND. I kept the header anyway: it is what
  tells the generator which company and year a passage belongs to, which the final accuracy depends
  on, and V7/V9 recover the crowding. But the row is reported as measured — on the public set the
  header is not free.
- **V6, better tokenizer (recall unchanged, +1.7 accuracy points).** Stop words, possessive stripping
  (`Brightwater Energy's` → `brightwater energy`), numbers without commas and plural folding swap one
  hit for another: q027 (headquarters city, lost at V5) is recovered through the synonym
  `headquartered` → `headquarters`, and q013 (a comparison) becomes a miss, so recall stays at 45/46.
  Accuracy rises because one more table question (q024) is also answered correctly, and because the
  synonym map (`guide` → `outlook`, `headcount` → `employees`) is what the coverage term and the
  abstention threshold rely on. Without it a question phrased with a word the report never uses scores
  zero coverage and is wrongly rejected.
- **V7, per-company retrieval (recall 97.8% → 100%; comparisons to 100%).** A comparison needs two
  facts in two documents; one ranking over a mixed pool returns five chunks of whichever company
  scores higher. Ranking each company separately and taking its top chunk(s) guarantees both are
  represented. It also brings the mean context from 5.00 to 4.80 chunks.
- **V8, abstention (rejected 0% → 100%, Task 3 from 0.0 to 10.0 points).** Two hard rules cover 10 of
  the 14 unanswerable questions with no effect on the answerable ones: a company that is not in the
  corpus, and a year that the company's documents of that type do not state. The other four are
  undisclosed facts (lobbying expenses, churn, carbon offset purchases, revenue from the largest
  customer). A phrase list catches those phrasings before any model call; a YES/NO check on the top
  three chunks is the fallback for an undisclosed fact phrased in words the list does not know, and it
  runs about 19 times per run. The phrase list is the weakest part of this design: it is written
  against the phrasings of the public questions, so a private question about a different undisclosed
  figure falls through to the model check, which is the weaker of the two.
- **V9, compact context (recall unchanged, +1.7 accuracy points, 4.80 → 2.47 chunks).** Dedupe (one
  chunk per table/section), a cap of three chunks, and dropping the tail below 35% of the best score
  roughly halve the context. Recall is unchanged because the right chunk was already near the top;
  what changes is that the generator sees fewer look-alike passages, and on the public set that is
  enough to turn the one remaining wrong answer into a correct one. A one-question gain is within
  noise, so the stronger argument for this row is the shorter, cheaper context at no recall cost.

### Error analysis

What the remaining misses are, row by row, and why the ladder has two small dips:

1. **Under B0, 33 of 46 answerable questions miss, in three families.** 19 are table questions:
   BM25 over the whole corpus cannot tell one company's or one year's statements from another's, so
   the five chunks rarely include the right one. 11 of the 12 comparisons miss, because they need two
   documents and a single ranking almost never supplies both. 3 are fact questions. B0 still answers
   43.3% correctly; part of that is the generator saying NOT_FOUND to unanswerable questions when the
   context does not fit, so accuracy flatters B0 and recall is the better measure of its retrieval.
2. **The V1/V2 dips are real and only partly explained.** Under plain BM25 over the whole corpus,
   both changes lose a few hits (35 and 36 misses against 33). My working explanation is that they
   change chunk boundaries and lengths in ways BM25's length normalisation does not favour, but I did
   not isolate this further. What is measured is that the dips disappear once the pool is filtered
   and the scoring is fixed, and that both changes are needed for the final rows (the generator reads
   row labels and column headers; filtering and dedupe work on whole tables). I kept the ladder in
   this order rather than reorder it to hide the dips.
3. **What V3 still misses.** 14 misses remain after the pool is filtered, all of the two families
   above: the cover page outranks the statements (11 table questions) and a comparison still gets five
   chunks of one company (3). This row shows that metadata alone is not enough — it narrows the pool
   but does not rank inside it.
4. **What the last rows fix and what they cost.** V5's header costs one narrative hit (q027) and one
   unanswerable question (q044); V6 recovers q027 but loses q013; V7 recovers q013 and makes every
   comparison a hit; V9 fixes the last wrong answer by shrinking the context. The V5 regression is the
   instructive case: adding metadata to the chunk text also adds it to *every* chunk of a report, so
   the company name and year stop distinguishing one chunk of that report from another and the section
   title has to carry the ranking.

Development bugs found while building this, kept because the same mistakes are easy to repeat:

- **The year regex missed every `FY2025`.** `\b(20\d{2})\b` does not match inside "FY2025" because
  `Y` and `2` are both word characters, so 21 of 60 questions parsed `fiscal_year: None`, their year
  filter was skipped, and half the answerable questions came back empty. `(?<!\d)(20\d{2})(?!\d)`
  took the run from 44 to 66 points.
- **`"rating"` matched inside `"operating"`.** Document-type hints were substrings, so every question
  about *operating* income was classified as a research-report question and its pool became the
  broker note. Word-boundary regexes fixed it; Task 2.2 went to 100%.
- **Possessives broke query expansion.** The tokenizer produced `software's` and `software'`, which
  appear nowhere in the corpus, so the coverage of an otherwise perfect chunk fell to 8% and the
  answerability check rejected it. Stripping the possessive before tokenising took the score from 66
  to 72.
