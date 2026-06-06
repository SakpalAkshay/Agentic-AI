# Conditional Workflows in LangGraph (Revision Notes)

---

## Table of Contents

1. [What is a Conditional Workflow?](#1-what-is-a-conditional-workflow)
2. [Conditional vs Parallel — The Critical Difference](#2-conditional-vs-parallel--the-critical-difference)
3. [The Conditional Edge Pattern](#3-the-conditional-edge-pattern)
4. [Example 1 — Quadratic Equation Solver (Non-LLM)](#4-example-1--quadratic-equation-solver-non-llm)
5. [Example 2 — Customer Review Reply (LLM-Based)](#5-example-2--customer-review-reply-llm-based)
6. [Quick Revision Summary](#6-quick-revision-summary)

---

## 1. What is a Conditional Workflow?

A **conditional workflow** is one where, at some point in the flow, control goes **to one branch OR another** — not both — based on a **condition**.

It looks similar to a parallel workflow visually (there are multiple branches), but only **one branch is followed** per run.

> Think of it as the workflow equivalent of `if/else` in programming. There can be 2, 3, or more branches — no limit.

> **Importance:** Going forward, when building more complex workflows, you will need conditional branching **almost every time**. It is as important to LangGraph as `if/else` is to programming.

---

## 2. Conditional vs Parallel — The Critical Difference

Both have branches, but they behave fundamentally differently:

| | Parallel Workflow | Conditional Workflow |
|---|-------------------|----------------------|
| Branches taken | **All** simultaneously | **One** (based on a condition) |
| Execution order | All branches run together, then fan in | Task 1 → (Task 2 OR Task 3) → Task 4 |
| Visual cue | Solid edges to multiple nodes | **Dotted/dashed** edges (in LangGraph viz) |

In a parallel workflow with Task 1 → {Task 2, Task 3} → Task 4: Tasks 2 and 3 always run together. In a conditional workflow, **either** 2 runs, **or** 3 runs — never both.

---

## 3. The Conditional Edge Pattern

To build a conditional edge in LangGraph you need:

1. **A condition-checking function** — a separate Python function that takes the state, evaluates a condition, and **returns the name of the next node** to go to.
2. **`graph.add_conditional_edges(...)`** instead of `graph.add_edge(...)`. You pass it the source node and the condition function.

The condition function is *not itself a node* — it is a **routing function** that decides which node to execute next.

```python
def check_condition(state) -> Literal['node_a', 'node_b', 'node_c']:
    if state['value'] > 0:
        return 'node_a'
    elif state['value'] == 0:
        return 'node_b'
    else:
        return 'node_c'

# In the graph:
graph.add_conditional_edges('source_node', check_condition)
```

> There is also a second way to create conditional edges using a `command` function — covered later when building dynamic workflows. This video covers the first way.

---

## 4. Example 1 — Quadratic Equation Solver (Non-LLM)

**The problem:** Given a quadratic equation `ax² + bx + c = 0`, find its roots. The solution path depends on the **discriminant** `D = b² - 4ac`:

| Condition | Outcome |
|-----------|---------|
| `D > 0` | **Two distinct real roots**: `(-b ± √D) / 2a` |
| `D == 0` | **One repeated root**: `-b / 2a` |
| `D < 0` | **No real roots** |

This naturally maps to a conditional workflow.

**The workflow:**

```
START → show_equation → calculate_discriminant → ?
                                                 ├─ (D > 0)  → real_roots     → END
                                                 ├─ (D == 0) → repeated_roots → END
                                                 └─ (D < 0)  → no_real_roots  → END
```

**The state:** `a`, `b`, `c` (inputs as int), `equation` (str), `discriminant` (float), `result` (str).

Create a file `quadratic_equation.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


# ---------- Define the State ----------
class QuadState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str


# ---------- Node 1: Show the Equation ----------
def show_equation(state: QuadState):
    equation = f'{state["a"]}x² + {state["b"]}x + {state["c"]}'
    return {'equation': equation}


# ---------- Node 2: Calculate the Discriminant ----------
def calculate_discriminant(state: QuadState):
    discriminant = (state['b'] ** 2) - (4 * state['a'] * state['c'])
    return {'discriminant': discriminant}


# ---------- Node 3: Two Real Roots (D > 0) ----------
def real_roots(state: QuadState):
    root1 = (-state['b'] + state['discriminant'] ** 0.5) / (2 * state['a'])
    root2 = (-state['b'] - state['discriminant'] ** 0.5) / (2 * state['a'])
    result = f'The roots are {root1} and {root2}'
    return {'result': result}


# ---------- Node 4: One Repeated Root (D == 0) ----------
def repeated_roots(state: QuadState):
    root = -state['b'] / (2 * state['a'])
    result = f'Only repeating root is {root}'
    return {'result': result}


# ---------- Node 5: No Real Roots (D < 0) ----------
def no_real_roots(state: QuadState):
    result = 'No real roots'
    return {'result': result}


# ---------- The condition-checking (routing) function ----------
# Notice: NOT a node. It returns the NAME of the next node to execute.
def check_condition(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'


# ---------- Build the Graph ----------
graph = StateGraph(QuadState)

graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

# Normal edges
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

# Conditional edge — this is the key new piece
graph.add_conditional_edges('calculate_discriminant', check_condition)

# All three branches converge to END
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

workflow = graph.compile()

# ---------- Execute ----------
initial_state = {'a': 4, 'b': -5, 'c': -4}
final_state = workflow.invoke(initial_state)
print(final_state)
```

> When you visualize this graph, the edges leaving `calculate_discriminant` are drawn as **dotted lines** — that is LangGraph's visual cue for **conditional edges**. Only one of them will execute per run.

**How the routing function works:** After `calculate_discriminant` runs, LangGraph calls `check_condition(state)`. It returns a node name as a string (e.g., `'real_roots'`); LangGraph then routes execution to that node. Behind the scenes that string is matched against the registered node names.

> **Core takeaway:** Build a function that **checks a condition** and **returns the next node name**, then use **`add_conditional_edges`** instead of `add_edge` — LangGraph handles the rest.

---

## 5. Example 2 — Customer Review Reply (LLM-Based)

**The application:** A customer leaves a review. The system must reply — but **how** it replies depends on the **sentiment**:

- **Positive sentiment** → send a warm thank-you reply.
- **Negative sentiment** → first **diagnose** the review (find `issue_type`, `tone`, `urgency`), then write an **empathetic resolution** reply that addresses all three.

**The workflow:**

```
START → find_sentiment → ?
                         ├─ (positive) → positive_response → END
                         └─ (negative) → run_diagnosis → negative_response → END
```

**Why this needs more than the first example:** It combines **conditional branching** with **structured output** (for the sentiment and for the diagnosis).

**The state:** `review`, `sentiment` (a `Literal['positive', 'negative']`), `diagnosis` (a dict with `issue_type`, `tone`, `urgency`), `response`.

Create a file `customer_review.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4o-mini')


# ---------- Structured Schema 1: for sentiment classification ----------
class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Sentiment of the review')


# ---------- Structured Schema 2: for diagnosing negative reviews ----------
class DiagnosisSchema(BaseModel):
    issue_type: Literal['UX', 'Performance', 'Bug', 'Support', 'Other'] = Field(
        description='The category of issue mentioned in the review')
    tone: Literal['angry', 'frustrated', 'disappointed', 'calm'] = Field(
        description='The emotional tone expressed by the user')
    urgency: Literal['low', 'medium', 'high'] = Field(
        description='How urgent or critical the issue appears to be')


# Bind both schemas to dedicated structured models
structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)


# ---------- Define the State ----------
class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response: str


# ---------- Node 1: Find the sentiment ----------
def find_sentiment(state: ReviewState):
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = structured_model.invoke(prompt).sentiment
    return {'sentiment': sentiment}


# ---------- Routing function: check sentiment ----------
def check_sentiment(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'


# ---------- Node 2: Positive Response ----------
def positive_response(state: ReviewState):
    prompt = f'''Write a warm thank-you message in response to this review:
{state["review"]}
Also, kindly ask the user to leave feedback on our website.'''
    # Use the NORMAL model (no structured output needed here)
    response = model.invoke(prompt).content
    return {'response': response}


# ---------- Node 3: Run Diagnosis (for negative reviews) ----------
def run_diagnosis(state: ReviewState):
    prompt = f'''Diagnose this negative review:
{state["review"]}
Return issue_type, tone and urgency.'''
    response = structured_model2.invoke(prompt)
    # Convert the Pydantic object into a dict (so it fits the 'diagnosis' state key)
    return {'diagnosis': response.model_dump()}


# ---------- Node 4: Negative Response ----------
def negative_response(state: ReviewState):
    diagnosis = state['diagnosis']
    prompt = f'''You are a support assistant. The user had a {diagnosis["issue_type"]} issue,
sounded {diagnosis["tone"]}, and marked urgency as {diagnosis["urgency"]}.
Write an empathetic, helpful resolution message.'''
    response = model.invoke(prompt).content
    return {'response': response}


# ---------- Build the Graph ----------
graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

# Edges
graph.add_edge(START, 'find_sentiment')

# Conditional edge from find_sentiment
graph.add_conditional_edges('find_sentiment', check_sentiment)

# Positive branch goes straight to END
graph.add_edge('positive_response', END)

# Negative branch: run_diagnosis → negative_response → END
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()

# ---------- Execute ----------
# Example with a negative review
initial_state = {
    'review': "I've been trying to log in for over an hour now, and the app keeps "
              "freezing on the authentication screen. I even tried reinstalling it, "
              "but no luck. This kind of bug is unacceptable, especially when it "
              "affects basic functionality."
}
final_state = workflow.invoke(initial_state)
print(final_state)
```

**For the negative example above, the output looks roughly like:**
- `sentiment` → `'negative'`
- `diagnosis` → `{'issue_type': 'Bug', 'tone': 'frustrated', 'urgency': 'high'}`
- `response` → an empathetic resolution message addressing all three diagnosis fields.

**Two things to notice in this example:**

1. **Two structured models** are used — one for sentiment (`SentimentSchema`) and one for the diagnosis (`DiagnosisSchema`). They are bound separately via `with_structured_output(...)`.
2. **`.model_dump()`** converts the Pydantic response object into a plain dict, so it can be stored in the `diagnosis` state field (which is typed as `dict`).

> Notice: only **`find_sentiment` and `positive_response`** OR only **`find_sentiment → run_diagnosis → negative_response`** will execute on any given run — never all four. That's the conditional behavior in action.

---

## 6. Quick Revision Summary

**Conditional workflow** = at some point, control goes to **one** of several branches based on a **condition** — like `if/else` in programming. Visualized in LangGraph as **dotted edges**.

**Conditional vs Parallel:**
- *Parallel:* all branches run together.
- *Conditional:* exactly one branch runs.

**How to build a conditional edge in LangGraph:**

| Step | Code |
|------|------|
| 1. Write a routing function | `def check_x(state) -> Literal['n1', 'n2']:` returns the **name** of the next node. |
| 2. Wire it up | `graph.add_conditional_edges('source_node', check_x)` instead of `add_edge`. |

The routing function is **not a node** — it just decides which node runs next.

**Two examples built:**

1. **Quadratic Equation Solver** (non-LLM): `show_equation → calculate_discriminant → ?` branches to `real_roots`, `repeated_roots`, or `no_real_roots` based on the discriminant.
2. **Customer Review Reply** (LLM-based, more involved): `find_sentiment → ?` either `positive_response → END` or `run_diagnosis → negative_response → END`. Uses **two Pydantic schemas** for structured output (sentiment + diagnosis), and `.model_dump()` to store a Pydantic result as a dict in the state.

**Key facts to remember:**
- The routing function returns a **node-name string**, which LangGraph matches against registered node names.
- Use **`add_conditional_edges(source, routing_fn)`** for conditional branching; use the regular `add_edge` for sequential connections that follow.
- Conditional edges show up as **dotted lines** in the visualized graph.
- For LLM structured output: define a Pydantic `BaseModel`, bind it with `model.with_structured_output(Schema)`, and access fields via `.field_name` (or `.model_dump()` for a dict).
- Use the **normal model** (not the structured one) when generating free-form text like a reply — a structured model would try to fill schema fields you don't want.
- There is **a second way** to build conditional edges using a `command` function — covered later when building dynamic workflows.

---

*Revision notes — Conditional Workflows in LangGraph (Playlist Video 7).*
