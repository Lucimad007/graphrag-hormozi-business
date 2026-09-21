from __future__ import annotations

STEP_BACK_SYSTEM = """\
Restate the user's business question, then write a step-back search question.

Return JSON only:
{"restated_query": "...", "step_back_query": "...", "situation": "..."}

restated_query (one sentence):
- Keep numbers, percents, dollar amounts, time ranges, and TRENDS (up, down, flat, from X to Y).
- Keep channel, offer type, industry, and constraints the user stated.
- Do not invent metrics or directions the user did not give.

step_back_query (one sentence, for vector search only):
- Ask the underlying playbook principle (offer, close, nurture, leads, pricing, guarantee).
- Drop proper names, platforms, and dollar amounts unless they ARE the concept (e.g. decoy offer).
- Keep the metric and its direction when that is the diagnosis (declining close rate, high churn).
- If the user already asked a principle-level question, copy restated_query exactly.

situation (one short line of facts, not a question):
- Only what the user stated: business type, channel, offer, metrics, TRENDS.
- Empty string if the question is already a pure definition/how-to with no case facts.

Examples:
"input": "Gym, lots of Instagram leads, close rate fell from 40% to 20% on a $3k coaching offer. What is going on?"
"output_restated": "Why did close rate drop from 40% to 20% on a $3k coaching offer despite high Instagram lead volume?"
"output_step_back": "What causes close rate to decline when lead volume stays high?"
"output_situation": "Gym coaching at $3k; Instagram lead volume high; close rate fell 40% to 20%."

"input": "Could the members of The Police perform lawful arrests?"
"output_restated": "Could members of The Police perform lawful arrests?"
"output_step_back": "What powers do police officers have?"
"output_situation": ""

"input": "How do decoy offers work in a money model?"
"output_restated": "How do decoy offers work in a money model?"
"output_step_back": "How do decoy offers work in a money model?"
"output_situation": ""
"""


GENERATE_ANSWER_SYSTEM = """\
Answer using only the evidence for the original situation. Cite document names.
The PDFs are playbooks. They will almost never name the user's gym, Instagram, or exact percents. That is expected.
Apply those playbooks to THIS situation (industry, channel, offer, numbers, trends in "situation").
Do not say evidence is insufficient merely because those specifics are missing from the sources.
Do not answer only the generic step-back question.
Lead with what is going on in their business, then ground each point in a cited document.
Do not invent facts the user did not state.
Return JSON {"answer": "..."} only.
"""


def normalize_search_query(text: str) -> str:
    return " ".join(text.split()).strip()


def unique_retrieval_queries(*parts: str | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        query = normalize_search_query(part or "")
        if not query:
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(query)
    return ordered
