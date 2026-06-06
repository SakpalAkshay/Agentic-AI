# Iterative (Loop-Based) Workflows in LangGraph (Revision Notes)
---

## Table of Contents

1. [What is an Iterative Workflow?](#1-what-is-an-iterative-workflow)
2. [The Real-World Use Case — Auto-Posting Tweets](#2-the-real-world-use-case--auto-posting-tweets)
3. [The Workflow Design — Generate / Evaluate / Optimize](#3-the-workflow-design--generate--evaluate--optimize)
4. [Setting Up — Three Specialized LLMs](#4-setting-up--three-specialized-llms)
5. [Defining the State](#5-defining-the-state)
6. [Building the Nodes](#6-building-the-nodes)
7. [Wiring the Loop — Edges & the Conditional Router](#7-wiring-the-loop--edges--the-conditional-router)
8. [Running the Workflow](#8-running-the-workflow)
9. [Optional Enhancement — Tracking History](#9-optional-enhancement--tracking-history)
10. [Quick Revision Summary](#10-quick-revision-summary)

---

## 1. What is an Iterative Workflow?

An **iterative** (also called **looping**) workflow is one where **two or more nodes form a loop** that runs repeatedly **in order to improve something** — until a stopping condition is met.

> This is a very important kind of workflow. As you build more complex systems going forward, you'll see iterative loops used in many places.

So far the playlist has covered four workflow types:

| Type | Behavior |
|------|----------|
| **Sequential** | Tasks run one after another, linearly |
| **Parallel** | Multiple tasks run at the same time |
| **Conditional** | Exactly one branch runs based on a condition |
| **Iterative** | Two or more nodes loop until something is "good enough" |

---

## 2. The Real-World Use Case — Auto-Posting Tweets

**The story:** A YouTuber wants to be active on other platforms (LinkedIn, X/Twitter, Instagram) too, but doesn't have the time to write quality posts manually. An automated workflow could post on his behalf — but the problem is **quality**: an LLM's *first* attempt at a tweet is often **mediocre** and provides little value to readers.

**The fix:** Use an **iterative workflow** — generate a tweet, **evaluate** it against strict criteria, and if it's not good enough, **optimize** it and re-evaluate. Loop until it's approved.

> Specifically, the workflow targets the **X (Twitter)** platform and generates **funny + original tweets** on a given topic.

---

## 3. The Workflow Design — Generate / Evaluate / Optimize

Three core components: a **Generator**, an **Evaluator**, and an **Optimizer**.

```
START → generate ──→ evaluate ──?──→ END                  (if approved)
                       ▲          │
                       │          └──→ optimize ──┐       (if needs improvement)
                       └──────────────────────────┘
```

**Step by step:**

1. **Generate** — The user provides a *topic*. The **Generator LLM** writes a tweet on that topic.
2. **Evaluate** — The generated tweet goes to the **Evaluator LLM**, which is given **strict criteria** describing what counts as a good tweet. It returns:
   - An **evaluation result**: `approved` or `needs_improvement`.
   - **Constructive feedback** explaining strengths/weaknesses.
3. **Decide** (conditional edge):
   - If `approved` → **END** (in a real product, this would go to a human-in-the-loop for final review, then post via an API).
   - If `needs_improvement` → go to **Optimize**.
4. **Optimize** — The **Optimizer LLM** receives the *current tweet* + the *evaluator's feedback* and produces an **improved tweet**. The new tweet goes **back to Evaluate**.
5. The loop repeats until either the tweet is approved **or** the maximum number of iterations is hit (so the workflow can't get stuck infinitely).

> The clearer and stricter your evaluation criteria, the better the final outcome. The video uses long, well-described prompts for both the evaluator and the optimizer.

---

## 4. Setting Up — Three Specialized LLMs

In an ideal real-world setup, the three LLMs should be **different** and chosen for what each is best at:

| LLM | Best at | Example choice |
|-----|---------|----------------|
| **Generator** | Strong creative writing | A powerful model (e.g., GPT-4.5) |
| **Evaluator** | Following strict instructions | Any reliable instruction-following model |
| **Optimizer** | Strong writing | Strong writing model |

For this demo all three are similar. The video uses `gpt-4o` for the generator and optimizer, and `gpt-4o-mini` for the evaluator (and later switches the generator to `gpt-4o-mini` deliberately to produce weaker tweets so the loop actually has to iterate).

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
import operator

load_dotenv()

generator_llm = ChatOpenAI(model='gpt-4o')
evaluator_llm = ChatOpenAI(model='gpt-4o-mini')
optimizer_llm = ChatOpenAI(model='gpt-4o')
```

> In your real project, research which model is best for each role and just swap it here — the rest of the code stays the same.

---

## 5. Defining the State

The state needs: the topic, the current tweet, the evaluation result (a `Literal`), the latest feedback, an iteration counter, and a max-iteration safety limit.

```python
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal['approved', 'needs_improvement']
    feedback: str
    iteration: int
    max_iteration: int
```

> **Why `max_iteration` matters:** Without a cap, the loop could run **infinitely** — the evaluator keeps rejecting, the optimizer keeps rewriting, and so on. This can happen if the criteria are too strict for the model. A hard limit (e.g., 5) breaks the loop safely. Pick a value that suits your project.

---

## 6. Building the Nodes

Three nodes: **`generate_tweet`**, **`evaluate_tweet`**, and **`optimize_tweet`**. The evaluator uses **structured output** (a Pydantic schema) so its response is always reliably parseable.

### Node 1 — Generate Tweet

Builds a `messages` list (a `SystemMessage` + `HumanMessage`, learned in the LangChain playlist) instead of an f-string, so the model behavior is more controllable.

```python
def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, and relatable takes.
- Use simple, day-to-day English.
""")
    ]

    response = generator_llm.invoke(messages).content
    return {'tweet': response}
```

### Node 2 — Evaluate Tweet (with structured output)

```python
# Pydantic schema for the structured evaluation response
class TweetEvaluation(BaseModel):
    evaluation: Literal['approved', 'needs_improvement'] = Field(
        ..., description="Final evaluation result.")
    feedback: str = Field(..., description="Feedback for the tweet.")


structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)


def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. "
                              "You evaluate tweets based on humor, originality, "
                              "virality, and tweet format."),
        HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it 100 times before?
2. Humor – Did it genuinely make you smile, laugh, or chuckle?
3. Punchiness – Is it short, sharp, and scroll-stopping?
4. Virality Potential – Would people retweet or share it?
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did...because...")
- It exceeds 280 characters
- It reads like a traditional joke
- It lacks a strong punchline or surprise

Respond only in structured format:
- evaluation: "approved" or "needs_improvement"
- feedback: One paragraph explaining strengths and weaknesses
""")
    ]

    response = structured_evaluator_llm.invoke(messages)

    return {'evaluation': response.evaluation, 'feedback': response.feedback}
```

### Node 3 — Optimize Tweet

Receives the current tweet + feedback and produces an improved version. Also **increments the iteration counter**.

```python
def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1     # increment the loop counter

    return {'tweet': response, 'iteration': iteration}
```

---

## 7. Wiring the Loop — Edges & the Conditional Router

The loop is built using a **conditional edge** from `evaluate` (decides whether to exit or keep optimizing) **plus** a regular edge from `optimize` back to `evaluate`.

The routing function exits the loop on **either** approval **or** hitting the max-iteration cap:

```python
def route_evaluation(state: TweetState) -> Literal['approved', 'needs_improvement']:
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'
```

Now wire it all together:

```python
graph = StateGraph(TweetState)

graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

# Linear part: START → generate → evaluate
graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

# Conditional branch from evaluate:
#   approved          → END
#   needs_improvement → optimize
graph.add_conditional_edges('evaluate', route_evaluation, {
    'approved': END,
    'needs_improvement': 'optimize'
})

# The loop-back edge: optimize → evaluate
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()
```

> **This is how you create loops in LangGraph.** You simply **manipulate the edges** — nothing more. A conditional edge controls the exit, and a normal edge sends control *back* to a previous node.

---

## 8. Running the Workflow

```python
initial_state = {
    'topic': 'Indian Railways',
    'iteration': 1,            # this is the first iteration
    'max_iteration': 5
}

result = workflow.invoke(initial_state)
print(result['tweet'])
```

Sometimes the first generated tweet is approved immediately; other times the loop runs 2 or 3 times before approval. If the workflow hits `max_iteration`, the latest tweet is returned anyway (the loop is forced to exit).

> **A debugging tip from the video:** if a tweet keeps getting approved on the first try and you want to see the loop iterate, deliberately use a weaker model for the generator (e.g., `gpt-4o-mini` instead of `gpt-4o`).

---

## 9. Optional Enhancement — Tracking History

To **see every intermediate tweet and feedback** the loop produced, store them in the state as a list — using a **reducer function** so each new value is **appended** (not replaced) when multiple iterations write to the same key.

**Add two history fields to the state** with the `operator.add` reducer:

```python
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal['approved', 'needs_improvement']
    feedback: str
    iteration: int
    max_iteration: int

    # NEW — history with the add reducer so writes APPEND rather than replace
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
```

**In `generate_tweet`** — also append the generated tweet:

```python
return {'tweet': response, 'tweet_history': [response]}
```

**In `optimize_tweet`** — also append the new tweet:

```python
return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}
```

**In `evaluate_tweet`** — also append the feedback:

```python
return {'evaluation': response.evaluation,
        'feedback': response.feedback,
        'feedback_history': [response.feedback]}
```

> Each node returns its value **inside a single-element list** (e.g., `[response]`); the `operator.add` reducer then merges them across iterations into one growing list — same pattern used in the parallel-workflows video for collecting parallel scores.

You can then inspect the full history:

```python
result = workflow.invoke(initial_state)
for tweet in result['tweet_history']:
    print(tweet)
    print('---')
```

---

## 10. Quick Revision Summary

**Iterative workflow** = two or more nodes form a **loop** that repeats until a stopping condition is met — typically used to **iteratively improve** an output. Four workflow types so far: sequential, parallel, conditional, **iterative**.

**The Generate / Evaluate / Optimize pattern:**
- **Generator** produces a draft.
- **Evaluator** judges against strict criteria → returns `approved`/`needs_improvement` + feedback. Uses **structured output** for reliability.
- **Optimizer** rewrites using the feedback → loops back to Evaluator.
- Exit when approved **or** the max-iteration cap is reached.

**How to build a loop in LangGraph — just edge manipulation:**

```python
# Conditional edge: exits or branches to the loop body
graph.add_conditional_edges('evaluate', route_evaluation, {
    'approved': END,
    'needs_improvement': 'optimize'
})

# Plain edge: the loop-back
graph.add_edge('optimize', 'evaluate')
```

**Key implementation facts:**
- The router function exits on `evaluation == 'approved'` **OR** `iteration >= max_iteration` — always include the **iteration cap** to prevent infinite loops.
- The optimizer node **increments the iteration counter** every time it runs.
- Use `SystemMessage` + `HumanMessage` lists (instead of f-strings) for cleaner prompt control — covered in the LangChain playlist.
- Structured output (Pydantic schema + `with_structured_output(...)`) ensures the evaluation is always parseable.
- For **history tracking**, use `Annotated[list[str], operator.add]` in the state and return each new value inside a `[list]` from the node — the reducer appends across iterations.

**Workflow in this video:**
- A Twitter/X post generator targeting **funny + original** tweets.
- `START → generate → evaluate → (approved → END | needs_improvement → optimize → back to evaluate)`.
- Capped at 5 iterations.

> In a real product, the "approved" branch would go to a **human-in-the-loop** for final review, then a **platform API call** to actually post. We'll improve this same workflow later when we learn about **tools** and **human-in-the-loop**.

---

*Revision notes — Iterative Workflows in LangGraph (Playlist Video 8).*
