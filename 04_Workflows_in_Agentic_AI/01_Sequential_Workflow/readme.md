# Sequential Workflows in LangGraph (Revision Notes)

---

## Table of Contents

1. [What is a Sequential Workflow?](#1-what-is-a-sequential-workflow)
2. [Installation & Setup](#2-installation--setup)
3. [The Standard LangGraph Coding Pattern](#3-the-standard-langgraph-coding-pattern)
4. [Example 1 — BMI Calculator (Non-LLM Workflow)](#4-example-1--bmi-calculator-non-llm-workflow)
5. [Example 1 Extended — Adding a Second Node](#5-example-1-extended--adding-a-second-node)
6. [Visualizing the Graph](#6-visualizing-the-graph)
7. [Example 2 — Simple LLM Workflow](#7-example-2--simple-llm-workflow)
8. [Example 3 — Prompt Chaining Workflow](#8-example-3--prompt-chaining-workflow)
9. [Key Takeaway — Why State is Powerful](#9-key-takeaway--why-state-is-powerful)
10. [Homework](#10-homework)
11. [Quick Revision Summary](#11-quick-revision-summary)

---

## 1. What is a Sequential Workflow?

A **sequential workflow** is a workflow where all tasks are connected in a **linear fashion** — after the first task you go to the second, after the second to the third, and so on.

- **No branching.**
- **No parallel paths.**
- Just a simple linear chain of tasks.

> LangGraph is technically an **overkill** for purely linear workflows (*"ghuma ke kaan pakadna"*). But building them first teaches the LangGraph coding syntax — the **true power** shows up later with complex workflows.

---

## 2. Installation & Setup

Create a project folder (e.g., `langgraph-tutorials`), open it in VS Code, then create and activate a **virtual environment**:

```bash
# Create a virtual environment named "myenv"
python -m venv myenv

# Activate it (Windows)
myenv\Scripts\activate
```

Install the required libraries:

```bash
pip install langgraph
pip install langchain
pip install langchain-openai
pip install python-dotenv
```

**Why each library:**

| Library | Purpose |
|---------|---------|
| `langgraph` | The framework for building the workflow graphs |
| `langchain` | LangGraph and LangChain work hand in hand — any **LLM-related component** (chat models, prompt templates, document loaders, text splitters) comes from LangChain |
| `langchain-openai` | To use **OpenAI's models** in the workflows |
| `python-dotenv` | To read **environment variables** (e.g., the API key) from a `.env` file |

**Test the installation** in a Jupyter notebook (`0_test_installation.ipynb`):

```python
from langgraph.graph import StateGraph
```

> All code in this video is written in **Jupyter notebooks** because LangGraph graphs can be easily **printed/visualized** there. Later, for full projects, normal `.py` files are used. (You may be prompted to install the `ipykernel` package — install it.)

---

## 3. The Standard LangGraph Coding Pattern

Every LangGraph workflow follows the same overall sequence of steps:

1. **Define the State** — a `TypedDict` class holding all the data points as key-value pairs.
2. **Define the Graph** — create a `StateGraph` object, passing in the state.
3. **Add Nodes** — `graph.add_node(...)`; each node is a **Python function** behind the scenes.
4. **Add Edges** — `graph.add_edge(...)`; connect the nodes (including `START` and `END`).
5. **Compile the Graph** — `graph.compile()`; checks the graph structure is logically correct.
6. **Execute the Graph** — `workflow.invoke(initial_state)`.

> Two facts to keep in mind for every node function: it **receives the state as input**, and it **returns the state as output** (after making a partial update).

---

## 4. Example 1 — BMI Calculator (Non-LLM Workflow)

The first workflow is deliberately **non-LLM** so the focus stays on **LangGraph syntax**.

**The workflow:** Take `height` and `weight` as input → pass them to a node that **calculates BMI** → show the result. A simple, linear, sequential workflow.

**The state** has three key-value pairs: `weight`, `height`, and `bmi`.

Create a file `bmi_workflow.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ---------- Step 1: Define the State ----------
# A TypedDict is a special dictionary where you declare the data type of each key.
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float


# ---------- Node function ----------
# A node, behind the scenes, is just a Python function.
# It receives the state as input and returns the state as output.
def calculate_bmi(state: BMIState) -> BMIState:
    # Extract values from the state
    weight = state['weight_kg']
    height = state['height_m']

    # BMI formula: weight / height^2
    bmi = weight / (height ** 2)

    # Partial update: write the result back into the state
    state['bmi'] = round(bmi, 2)

    # Return the (updated) state
    return state


# ---------- Step 2: Define the Graph ----------
graph = StateGraph(BMIState)   # always pass the state when creating the graph

# ---------- Step 3: Add Nodes ----------
# add_node(node_name, function_it_points_to)
# the node name and the function name can be different
graph.add_node('calculate_bmi', calculate_bmi)

# ---------- Step 4: Add Edges ----------
# START and END are dummy nodes marking where the graph begins/ends
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', END)

# ---------- Step 5: Compile the Graph ----------
workflow = graph.compile()

# ---------- Step 6: Execute the Graph ----------
initial_state = {'weight_kg': 80, 'height_m': 1.73}
final_state = workflow.invoke(initial_state)
print(final_state)
```

**How execution flows:** When you `invoke` the workflow with the initial state, those values populate the state. The first node activates, its function runs, makes a **partial update**, and returns the state. Since there's only one node, the workflow then stops. The `invoke` call returns the **final state object**.

> Always remember: you give the graph a state as **input**, and when execution finishes the graph returns a **state object** as output.

---

## 5. Example 1 Extended — Adding a Second Node

Now add a feature: based on the calculated BMI, also report whether the person is **underweight, normal, overweight, or obese**.

This means a **new node** in the workflow: `START → calculate_bmi → label_bmi → END`.

**Three changes** are needed:

**Change 1 — add a new field to the state:**

```python
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str        # <-- NEW field
```

**Change 2 — define the new node function:**

```python
def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']   # value calculated in the previous node

    # Decision-making based on the BMI value
    if bmi < 18.5:
        state['category'] = 'Underweight'
    elif 18.5 <= bmi < 25:
        state['category'] = 'Normal'
    elif 25 <= bmi < 30:
        state['category'] = 'Overweight'
    else:
        state['category'] = 'Obese'

    return state   # partial update + return
```

**Change 3 — register the node and add the new edge:**

```python
graph.add_node('label_bmi', label_bmi)

# Edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')   # new edge
graph.add_edge('label_bmi', END)               # END now comes after label_bmi
```

Re-compile and invoke as before — the returned final state now also contains the `category`.

---

## 6. Visualizing the Graph

LangGraph graphs can be **viewed visually** in a Jupyter notebook. The documentation provides a small snippet for this:

```python
from IPython.display import Image, display

display(Image(workflow.get_graph().draw_mermaid_png()))
```

> This is the reason the video uses Jupyter notebooks — this visualization code works in notebooks, not in plain `.py` files. It becomes very helpful once graphs get more complex.

---

## 7. Example 2 — Simple LLM Workflow

The simplest possible **LLM workflow**: `START → llm_qa → END`. The `llm_qa` node takes a question, asks the LLM, and writes the LLM's response into the state.

**The state:** two string attributes — `question` and `answer`.

Create a `.env` file with your OpenAI key, then a file `simple_llm_workflow.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

# Load the OpenAI API key from the .env file
load_dotenv()

# Create the model object (loads the default model)
model = ChatOpenAI()


# ---------- Define the State ----------
class LLMState(TypedDict):
    question: str
    answer: str


# ---------- Node function ----------
def llm_qa(state: LLMState) -> LLMState:
    # 1. Extract the question from the state
    question = state['question']

    # 2. Form a prompt
    prompt = f'Answer the following question: {question}'

    # 3. Ask the LLM
    answer = model.invoke(prompt).content   # .content holds the actual answer

    # 4. Update the answer in the state
    state['answer'] = answer

    return state


# ---------- Build the Graph ----------
graph = StateGraph(LLMState)

graph.add_node('llm_qa', llm_qa)

graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

workflow = graph.compile()

# ---------- Execute ----------
initial_state = {'question': 'How far is the moon from the earth?'}
final_state = workflow.invoke(initial_state)
print(final_state)

# To see only the answer:
print(final_state['answer'])
```

> When an LLM response comes back it contains many things; we only want the **`.content`** attribute, which holds the actual answer.

> This same result could be obtained with a single line — `model.invoke(prompt).content`. The point of the example is **not** the output but learning **how LangChain and LangGraph work hand in hand**.

---

## 8. Example 3 — Prompt Chaining Workflow

> **Prompt chaining** = making **multiple LLM calls in series**. The task is decomposed and completed through a sequence of LLM calls rather than one.

**The workflow:** Give the LLM a **topic** and have it generate a **blog**. Instead of going directly from topic to blog: first generate a **detailed outline** from the topic (LLM call 1), then generate the **blog** from the topic + outline (LLM call 2).

Structure: `START → create_outline → create_blog → END`. Two nodes, both interacting with the LLM — hence prompt chaining.

**The state:** `title`, `outline`, `content` (all strings).

Create a file `prompt_chaining.ipynb`:

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()


# ---------- Define the State ----------
class BlogState(TypedDict):
    title: str
    outline: str
    content: str


# ---------- Node 1: Create the outline ----------
def create_outline(state: BlogState) -> BlogState:
    title = state['title']

    prompt = f'Generate a detailed outline for a blog on the topic - {title}'
    outline = model.invoke(prompt).content

    state['outline'] = outline   # partial update
    return state


# ---------- Node 2: Create the blog ----------
def create_blog(state: BlogState) -> BlogState:
    title = state['title']
    outline = state['outline']

    prompt = f'Write a detailed blog on the title - {title} using the following outline \n {outline}'
    content = model.invoke(prompt).content

    state['content'] = content   # partial update
    return state


# ---------- Build the Graph ----------
graph = StateGraph(BlogState)

graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()

# ---------- Execute ----------
initial_state = {'title': 'Rise of AI in India'}
final_state = workflow.invoke(initial_state)

print(final_state)
print(final_state['outline'])   # see just the outline
print(final_state['content'])   # see just the blog
```

Two LLM calls happen, so this takes a bit longer to run.

---

## 9. Key Takeaway — Why State is Powerful

In the prompt chaining example, the final state gives you the **title**, the **outline**, AND the **content** — all three are accessible.

> If you built this same thing with **LangChain chains**, the final output would give you **only the blog content** — not the outline. This was a real problem faced with chains.

In LangGraph the **state** carries everything from start to end and **evolves** along the way, so every intermediate result remains accessible at the end. This is a direct benefit of the **state concept**.

---

## 10. Homework

Extend the prompt chaining workflow by **adding a third node** — an **evaluate node**:

- Currently there are two nodes: `create_outline` and `create_blog`.
- Add a third node whose prompt is roughly: *"Based on this outline, rate my blog"* and have it **generate an integer score**.
- You will need to **change the state** (add a score field) and **change the workflow** (add the node and edges).

> It's not a big improvement, but doing these small things yourself on a first attempt is **significant progress**.

---

## 11. Quick Revision Summary

**Sequential workflow** = a **linear** chain of tasks — no branching, no parallel paths. LangGraph is overkill for these alone, but they teach the syntax.

**Setup:** create a virtual env, then `pip install langgraph langchain langchain-openai python-dotenv`. `langchain` is needed because LLM components (chat models, prompts, loaders, splitters) come from it. Work in **Jupyter notebooks** so graphs can be visualized.

**The standard 6-step LangGraph pattern:**

| Step | Code |
|------|------|
| 1. Define State | `class MyState(TypedDict): ...` |
| 2. Define Graph | `graph = StateGraph(MyState)` |
| 3. Add Nodes | `graph.add_node('name', function)` |
| 4. Add Edges | `graph.add_edge(START, 'name')`, ... `graph.add_edge('name', END)` |
| 5. Compile | `workflow = graph.compile()` |
| 6. Execute | `workflow.invoke(initial_state)` |

**Key facts:**
- A **node** is just a **Python function**; the node name and function name can differ.
- Every node function **takes the state as input** and **returns the state as output** after a **partial update**.
- `START` and `END` are **dummy nodes** marking the graph's beginning and end.
- `compile()` checks the graph structure is logically valid (e.g., no orphan nodes).
- `invoke(initial_state)` runs the graph and returns the **final state**.
- LLM responses: use **`.content`** to extract the actual answer.

**Three examples built:**
1. **BMI Calculator** — non-LLM, focuses on pure syntax; extended with a `label_bmi` node.
2. **Simple LLM Workflow** — one `llm_qa` node; shows LangChain + LangGraph working together.
3. **Prompt Chaining** — `create_outline → create_blog`; multiple LLM calls in series.

**Why state matters:** unlike LangChain chains (which return only the final output), LangGraph's **state** carries every intermediate result (title, outline, content) to the end — because state is shared and evolves through the whole workflow.

---

*Revision notes — Sequential Workflows in LangGraph (Playlist Video 5).*
