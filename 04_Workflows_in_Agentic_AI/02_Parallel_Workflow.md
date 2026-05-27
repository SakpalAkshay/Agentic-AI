# Parallel Workflows in LangGraph (Revision Notes)

---

## Table of Contents

1. [What is a Parallel Workflow?](#1-what-is-a-parallel-workflow)
2. [Example 1 — Batsman Stats (Non-LLM Parallel Workflow)](#2-example-1--batsman-stats-non-llm-parallel-workflow)
3. [The `InvalidUpdateError` Problem & Partial State Updates](#3-the-invalidupdateerror-problem--partial-state-updates)
4. [Example 2 — UPSC Essay Evaluator (LLM-Based Parallel Workflow)](#4-example-2--upsc-essay-evaluator-llm-based-parallel-workflow)
5. [Structured Output](#5-structured-output)
6. [Reducer Functions](#6-reducer-functions)
7. [Building the UPSC Essay Evaluator — Full Code](#7-building-the-upsc-essay-evaluator--full-code)
8. [Sequential vs Parallel — Which Update Style to Use?](#8-sequential-vs-parallel--which-update-style-to-use)
9. [Quick Revision Summary](#9-quick-revision-summary)

---

## 1. What is a Parallel Workflow?

A **parallel workflow** is one where, after a node, **multiple nodes execute simultaneously** rather than one after another. The results of those parallel nodes are then **merged** to produce a final outcome.

This video builds two examples:
1. A **simple, non-LLM** parallel workflow (batsman cricket stats) — to get the basic idea.
2. A **more involved, LLM-based** parallel workflow (UPSC essay evaluator) — which also uses **structured output** and **reducer functions**.

---

## 2. Example 1 — Batsman Stats (Non-LLM Parallel Workflow)

**The application:** Given a batsman's innings data — **runs**, **balls**, **fours**, **sixes** — calculate three things:

| Output | Formula |
|--------|---------|
| **Strike Rate** | `(runs / balls) * 100` |
| **Boundary %** | `(fours*4 + sixes*6) / runs * 100` — what % of runs came from boundaries |
| **Balls Per Boundary (BPB)** | `balls / (fours + sixes)` — after how many balls the batsman hits a boundary |

> **Key observation:** all three quantities depend only on the input data and **none depends on another** — so they can be calculated **in parallel**.

**The workflow:** `START → [calculate_strike_rate, calculate_boundary_percent, calculate_bpb] (parallel) → summary → END`. The `summary` node merges the three quantities into a text summary.

**The state** has 8 fields: `runs`, `balls`, `fours`, `sixes` (inputs), `strike_rate`, `boundary_percent`, `bpb` (calculated), and `summary`.

Create a file `batsman_workflow.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ---------- Step 1: Define the State ----------
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    bpb: float
    boundary_percent: float
    summary: str


# ---------- Node 1: Strike Rate ----------
def calculate_strike_rate(state: BatsmanState):
    strike_rate = (state['runs'] / state['balls']) * 100
    # NOTE: partial update — return only the key this node computes
    return {'strike_rate': strike_rate}


# ---------- Node 2: Balls Per Boundary ----------
def calculate_bpb(state: BatsmanState):
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb': bpb}


# ---------- Node 3: Boundary Percent ----------
def calculate_boundary_percent(state: BatsmanState):
    boundary_percent = ((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs'] * 100
    return {'boundary_percent': boundary_percent}


# ---------- Node 4: Summary ----------
def summary(state: BatsmanState):
    summary = f"""
Strike Rate - {state['strike_rate']}
Balls per boundary - {state['bpb']}
Boundary percent - {state['boundary_percent']}
"""
    return {'summary': summary}


# ---------- Build the Graph ----------
graph = StateGraph(BatsmanState)

graph.add_node('calculate_strike_rate', calculate_strike_rate)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# ---- Edges: the parallel part ----
# START fans out to three nodes
graph.add_edge(START, 'calculate_strike_rate')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

# The three nodes fan in to summary
graph.add_edge('calculate_strike_rate', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

# ---------- Execute ----------
initial_state = {'runs': 100, 'balls': 50, 'fours': 6, 'sixes': 4}
final_state = workflow.invoke(initial_state)
print(final_state)
```

> Once you understand the **nodes and edges** concept, building parallel workflows is just as logical and visual as building sequential ones — only the edge wiring changes (START fans out, then nodes fan in).

---

## 3. The `InvalidUpdateError` Problem & Partial State Updates

If each parallel node **returns the entire state**, running the workflow produces an error:

```
InvalidUpdateError: At key 'runs': Can receive only one value per step.
```

**Why this happens:** In Example 1, the three parallel nodes (`strike_rate`, `boundary_percent`, `bpb`) each returned the **whole state object**. Even though they only *read* `runs` (and never modified it), returning the full state makes LangGraph think **all three nodes updated `runs`** (and `balls`, `fours`, `sixes`) at the same time. LangGraph cannot accept three simultaneous updates to the same key — it doesn't know which value is correct, so it raises a **conflict error**.

> This problem **does not appear in sequential workflows** (where nodes run one at a time), only in parallel ones.

**The solution — partial state updates:** Instead of returning the entire state, each node returns **only the key(s) it actually computes**, as a dictionary:

```python
# WRONG (for parallel) — returns the whole state, causes conflict
def calculate_strike_rate(state: BatsmanState):
    state['strike_rate'] = (state['runs'] / state['balls']) * 100
    return state

# RIGHT — return only the computed key
def calculate_strike_rate(state: BatsmanState):
    strike_rate = (state['runs'] / state['balls']) * 100
    return {'strike_rate': strike_rate}
```

> **Important fact:** node functions take a dictionary as input (the state *is* a dictionary) and can **return a dictionary as output** — it does not have to be the full state. Returning just the relevant key-value pair is called a **partial update**, and it is totally allowed.

(The code in Example 1 above already uses this correct partial-update style.)

---

## 4. Example 2 — UPSC Essay Evaluator (LLM-Based Parallel Workflow)

**The application:** A website where UPSC aspirants write an essay and receive feedback evaluated from multiple aspects.

**The workflow:** `START → essay text` → evaluate the essay in **parallel** on three aspects:
1. **Clarity of Thought**
2. **Depth of Analysis**
3. **Language**

Each aspect is evaluated by an **LLM call**, and each LLM returns **two things**: a **textual feedback** and a **score between 0 and 10**.

The three nodes feed into a **final evaluation node**, which does two jobs:
1. **Merge** the three text feedbacks into one **summarized feedback** (via an LLM).
2. **Average** the three scores into a **final average score**.

The output is two things: a **summarized feedback** and a **final average score**.

This example involves **three concepts together**:
1. How to make **parallel workflows**
2. How to get **structured output**
3. How to use **reducer functions**

---

## 5. Structured Output

**The problem:** Each evaluation node must return a textual feedback AND a number (0–10). If you just ask the LLM via a plain prompt, ~8 out of 10 times it works, but sometimes it returns the score as the word `"seven"` instead of `7` — and you can't compute an average from a word. The output **must be structured, consistent, and reliable**.

**The solution — structured output:** Define a **schema** in advance using **Pydantic** and tell the model to produce output strictly in that format.

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Use a model that supports structured output, e.g. gpt-4o-mini
model = ChatOpenAI(model='gpt-4o-mini')


# ---------- Define the schema with Pydantic ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)


# Bind the schema to the model
structured_model = model.with_structured_output(EvaluationSchema)
```

Now `structured_model.invoke(prompt)` returns an object matching the schema. You access `.feedback` and `.score` on it directly:

```python
result = structured_model.invoke(prompt)
print(result.feedback)   # the textual feedback
print(result.score)      # the integer score, e.g. 8
```

> A more descriptive schema (good `Field` descriptions) helps the LLM produce better-matched output. This concept was covered in detail in the LangChain playlist — review it there if it feels new.

---

## 6. Reducer Functions

**Why a reducer is needed here:** The three parallel nodes each produce a score, and all three scores must go into the **same** state attribute — `individual_scores` (a list). But because the three writes happen **in parallel**, the default behavior (**replace/update**) would mean only one score survives. To collect all three, the writes must be **merged** — and merging requires a **reducer function**.

**How it works:** Build the graph so each evaluation node returns its score **inside a list** (e.g., `[8]`, `[7]`, `[6]`). LangGraph then has three single-element lists for the same key. To combine them, you need the `+` operator on lists — `[8] + [7] + [6]` → `[8, 7, 6]`.

You attach the reducer to the state field using `Annotated`:

```python
from typing import TypedDict, Annotated
import operator

# operator.add is the functional equivalent of the + operator.
# It is used as the reducer so parallel writes to the list get APPENDED, not replaced.
class UPSCState(TypedDict):
    individual_scores: Annotated[list[int], operator.add]
    # ... other fields
```

> `operator` is a Python module containing the functional equivalents of operators. `operator.add` works like `+`. It is used as the reducer because you cannot write `+` directly as a reducer. Other reducer functions exist too (e.g., `max`, `min`).

---

## 7. Building the UPSC Essay Evaluator — Full Code

**The state** has: `essay` text, three textual feedbacks (`language_feedback`, `analysis_feedback`, `clarity_feedback`), an `overall_feedback`, an `individual_scores` list (with the `operator.add` reducer), and an `avg_score`.

Create a file `upsc_essay_workflow.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator

load_dotenv()

# A model that supports structured output
model = ChatOpenAI(model='gpt-4o-mini')


# ---------- Structured output schema ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the essay')
    score: int = Field(description='Score out of 10', ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)


# ---------- Define the State ----------
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    # reducer 'operator.add' so parallel score writes APPEND into the list
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


# ---------- Node 1: Evaluate Language ----------
def evaluate_language(state: UPSCState):
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    # partial update — return only relevant keys
    return {'language_feedback': output.feedback,
            'individual_scores': [output.score]}   # score inside a list


# ---------- Node 2: Evaluate Depth of Analysis ----------
def evaluate_analysis(state: UPSCState):
    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'analysis_feedback': output.feedback,
            'individual_scores': [output.score]}


# ---------- Node 3: Evaluate Clarity of Thought ----------
def evaluate_thought(state: UPSCState):
    prompt = f'Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback,
            'individual_scores': [output.score]}


# ---------- Node 4: Final Evaluation ----------
def final_evaluation(state: UPSCState):
    # 1. Summarize the three feedbacks (use the NORMAL model, not structured —
    #    a structured model might generate an extra score)
    prompt = f'''Based on the following feedbacks create a summarized feedback
Language feedback - {state["language_feedback"]}
Depth of analysis feedback - {state["analysis_feedback"]}
Clarity of thought feedback - {state["clarity_feedback"]}'''
    overall_feedback = model.invoke(prompt).content

    # 2. Average the individual scores
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}


# ---------- Build the Graph ----------
graph = StateGraph(UPSCState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# START fans out to the three evaluation nodes
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

# The three nodes fan in to final_evaluation
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()

# ---------- Execute ----------
essay = "..."   # the essay text to evaluate
initial_state = {'essay': essay}
final_state = workflow.invoke(initial_state)
print(final_state)
```

The returned final state contains `language_feedback`, `analysis_feedback`, `clarity_feedback`, `overall_feedback`, the `individual_scores` list (e.g., `[7, 8, 8]`), and the `avg_score`.

> Note in `final_evaluation` the summarizing call uses the **normal model**, not the structured one — a structured model would try to generate another score, which isn't wanted there.

---

## 8. Sequential vs Parallel — Which Update Style to Use?

In the sequential workflows video, nodes **returned the entire state**. Here, parallel nodes must do **partial state updates**. So which should you use?

| Workflow type | What works |
|---------------|------------|
| **Sequential** | Returning the full state works fine |
| **Parallel** | You **must** do partial updates (return only the computed keys) |
| **Both** | **Partial updates work everywhere** |

> **Recommendation:** Going forward, **always use partial state updates** — whether sequential or parallel. It's the one approach that works in both cases.

---

## 9. Quick Revision Summary

**Parallel workflow** = after a node, multiple nodes run **simultaneously**; their results are then **merged**. In LangGraph, `START` **fans out** to several nodes via multiple `add_edge` calls, and those nodes **fan in** to a common node.

**Two examples built:**
1. **Batsman Stats** (non-LLM) — strike rate, boundary %, and balls-per-boundary computed in parallel, then summarized.
2. **UPSC Essay Evaluator** (LLM-based) — essay evaluated in parallel on clarity, depth, and language; then a final node summarizes feedback and averages scores.

**Three key concepts from this video:**

| Concept | What it solves |
|---------|----------------|
| **Partial state updates** | In parallel workflows, returning the *full* state causes `InvalidUpdateError` ("can receive only one value per step") because LangGraph thinks every node updated every key. Fix: each node returns **only the keys it computes**, as a dict. |
| **Structured output** | LLM scores must be reliable (`7`, not `"seven"`). Define a **Pydantic schema** (`BaseModel` + `Field`) and bind it with `model.with_structured_output(Schema)`. Access results via `.feedback`, `.score`. |
| **Reducer functions** | When parallel nodes write to the **same** state key, the default replace behavior loses data. Attach a reducer via `Annotated[list[int], operator.add]` so writes are **merged/appended** instead of replaced. |

**Key facts:**
- A node function can **return a dictionary** (partial update) — it need not return the full state object.
- `operator.add` (from the `operator` module) is the functional `+`, used as a list-merging reducer. Other reducers: `max`, `min`, etc.
- For structured output use a capable model like `gpt-4o-mini`.
- In a node that summarizes (not scores), use the **normal model**, not the structured one.
- **Best practice:** always use **partial state updates** — works for both sequential and parallel workflows.

---

*Revision notes — Parallel Workflows in LangGraph (Playlist Video 6).*
