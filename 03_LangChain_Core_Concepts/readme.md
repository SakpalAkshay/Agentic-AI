# LangGraph — Core Concepts (Revision Notes)

---

## Table of Contents

1. [Quick Revision — What is LangGraph?](#1-quick-revision--what-is-langgraph)
2. [LLM Workflows](#2-llm-workflows)
   - [Common Workflow 1 — Prompt Chaining](#common-workflow-1--prompt-chaining)
   - [Common Workflow 2 — Routing](#common-workflow-2--routing)
   - [Common Workflow 3 — Parallelization](#common-workflow-3--parallelization)
   - [Common Workflow 4 — Orchestrator-Worker](#common-workflow-4--orchestrator-worker)
   - [Common Workflow 5 — Evaluator-Optimizer](#common-workflow-5--evaluator-optimizer)
3. [Graphs, Nodes, and Edges](#3-graphs-nodes-and-edges)
4. [State](#4-state)
5. [Reducers](#5-reducers)
6. [LangGraph's Execution Model](#6-langgraphs-execution-model)
7. [Quick Revision Summary](#7-quick-revision-summary)

---

## 1. Quick Revision — What is LangGraph?

In simple terms, **LangGraph is an orchestration framework**. When you give LangGraph an LLM workflow to execute, it first **represents that workflow as a graph**, where:

- **Every node** of the graph is a **task** of your workflow. (LLM workflow tasks can include calling an LLM, calling a tool, or making a decision.)
- **All nodes are connected via edges**, and these **edges** tell us which task should execute **next** after a given node.

> **Definition:** LangGraph is an **orchestration framework** for building **intelligent, stateful, and multi-step LLM workflows**.

Once the graph is built, you simply provide **input to the first node** and trigger the graph — all nodes then execute automatically in the right order, and the workflow completes.

LangGraph is **not restricted to just building graphs** — it provides additional features:

- **Parallel execution** — multiple nodes can run together after a node.
- **Loops / cycles** — control can return to a previous node.
- **Branching** — based on a condition, one node or another executes.
- **Memory** — record the executed tasks and conversations.
- **Resumability** — if the workflow breaks down at some task, you can resume from that exact point.

> Together, these core features make LangGraph an **ideal candidate for building Agentic and production-grade AI applications**.

---

## 2. LLM Workflows

**What is a workflow?** A workflow is a **series of tasks executed in order to achieve a goal**. *Example: the automated hiring example — create JD → post it → shortlist → interview → onboard. Only when this whole series of steps runs in the right order does the hiring workflow complete.*

**What is an LLM workflow?** An LLM workflow is a workflow where **many tasks in the series depend on LLMs**. *In the automated hiring example, an LLM is needed to write the JD, possibly to shortlist, and possibly to conduct interviews.* In short, any workflow that **uses LLMs during its execution** is an LLM workflow.

> LLM workflows are **step-by-step processes** used to build complex LLM applications. Each step performs a distinct task such as **prompting, reasoning, tool calling, memory access, or decision making**. Workflows can be **linear, parallel, branched, or looped** — enabling complex behaviors like retries, multi-agent communication, and tool-augmented reasoning.

Every application has its **own** workflow, but there are **common workflows** seen repeatedly across applications. The video covers **five common LLM workflows**.

---

### Common Workflow 1 — Prompt Chaining

A very common workflow where you **call the LLM multiple times in sequence**.

**Example:** An app where the user gives a **topic** and you must produce a **detailed report**. You don't go directly from topic to report. Instead, you **break the task down**: first generate an **outline** from the topic (LLM call 1), then generate the **detailed report** from the outline (LLM call 2).

Use prompt chaining when you have a **complex task** you want to **divide into sub-tasks**. A benefit: you can place **checks between steps** to verify the process is working correctly. *Example: a check that the report should not exceed 5000 words — if it does, exit.*

---

### Common Workflow 2 — Routing

In routing, you **understand a task** and then **decide who will execute it**.

**Example — a customer support chatbot:** A customer query arrives. It could be a **technical**, **refund-related**, or **sales-related** query. The query goes to an LLM, which **decides** the query type and **routes** it to the most capable specialized LLM — refund queries to one LLM, technical doubts to a second, sales queries to a third.

> Here, the routing LLM acts as a **decision maker** — a **router** — choosing which downstream LLM is most capable of solving the query.

---

### Common Workflow 3 — Parallelization

You **break a given task into multiple sub-tasks**, execute **all sub-tasks simultaneously**, then **merge their results** into a final outcome.

**Example — a content moderation workflow for YouTube:** When a video is published, it must be checked from **multiple angles** before going live: (1) does it follow **community guidelines**, (2) does it contain **misinformation**, (3) does it contain **sexual content**. These three checks are **independent** — none depends on the result of another — so they can run **in parallel**.

The flow: take the video content → generate a transcript → send to three LLMs in parallel (one per check) → each sends its result to an **aggregator** → the aggregator decides whether to publish or flag the video.

---

### Common Workflow 4 — Orchestrator-Worker

Very similar to parallelization — you also divide a task into **multiple parallel sub-tasks**. **The only difference:** in orchestrator-worker you **do not know the nature of the sub-tasks in advance** — they are decided **dynamically**.

| | Parallelization | Orchestrator-Worker |
|---|-----------------|---------------------|
| Sub-tasks | **Predefined** (known in advance) | **Dynamic** (decided at runtime) |
| Example | The 3 YouTube checks are fixed | Sub-tasks vary with the input query |

**Example — a research assistant** that builds a detailed research report on a query. It must search multiple platforms and aggregate the information — but **where to search and what to search depends on the query**:

- A **scientific / technical** term → search **Google Scholar** (research papers).
- A **social phenomenon / political incident** → search **Google News**.

An **orchestrator LLM** analyzes the incoming query and **assigns a different task to each worker LLM** depending on the input. Things still run in parallel and results are aggregated at the end, but the **task nature is not known in advance**.

---

### Common Workflow 5 — Evaluator-Optimizer

An interesting workflow for tasks that **cannot be done perfectly in one attempt** — typically **creative work** like drafting an email, writing a blog, poem, or story, which requires **iteration**.

> Just like a real writer: write a first draft → review what's missing → turn that into feedback → write a second draft → repeat in a loop until the final good product emerges.

It uses **two LLMs**:

- **Generator LLM** — produces a **solution** (e.g., the blog).
- **Evaluator LLM** — given a **concrete evaluation criteria**, either **accepts** or **rejects** the solution; if it rejects, it also gives **feedback**.

If rejected, the generator uses the **feedback** to produce a new solution. This **loops** until the evaluator is **satisfied** and accepts the solution — then the loop breaks and you get the output.

---

## 3. Graphs, Nodes, and Edges

> Together, these three are the **single most important core concept** of LangGraph. LangGraph represents any LLM workflow as a **graph**, so you must understand how it does this.

**Example — a UPSC essay practice website.** A user lands on the site → the site generates an **essay topic** → the user writes and submits an **essay** → the site **evaluates** it from multiple perspectives and generates a **score** → if the score is above the cut-off, congratulate; if below, provide **feedback** and offer the option to **rewrite** the essay (which loops back through evaluation).

**How to build this in LangGraph:**

1. First convert the **high-level goal** into **actionable steps** on paper: generate topic → collect the student's essay → evaluate → aggregate the multi-perspective results → tell them if the essay was good or bad → give feedback → optionally let them rewrite.
2. Then represent this flow as a **graph** in LangGraph.

In the resulting graph (the UPSC example evaluates on three things — **clarity of thought**, **depth of analysis**, and **language/vocabulary/grammar/tone** — each scored out of 5, total **15**, with a **threshold of 10**), you see two things: **nodes** and **edges**.

**Nodes:**

- Each node represents a **single task** of your workflow.
- Behind the scenes, **every node is just a Python function** — nothing more. If you can write a Python function, you can create a node.
- So the graph LangGraph builds is essentially a **set of interconnected Python functions**.

**Edges:**

- Edges connect nodes and tell us **which node executes next** after a given node.

> **In short: nodes tell us *what* to do; edges tell us *when* to do it (when to execute a node).**

**Types of edges:**

| Edge type | What it does |
|-----------|--------------|
| **Sequential** | One node runs after another |
| **Parallel** | Multiple nodes execute at the same time |
| **Conditional** | Branching — flow goes one direction or another based on a condition |
| **Loop** | Control returns to a previous node |

> The benefit of the graph structure is that it lets you express **sequential, parallel, branching, and looping** flows — all because a graph naturally supports this freedom.

---

## 4. State

> A very important core concept.

**What is state?** Any LLM workflow needs certain **pieces of data** to guide it through execution. *In the UPSC example: the **essay text** the candidate writes is required throughout — you need it to evaluate. The **scores** are also data that later execution depends on, since the final score decides whether the essay is good or bad.*

This data has two key properties: (1) it is **required for execution**, and (2) it **evolves over time** as execution progresses. *The essay variable changes when the student rewrites; the score changes as evaluation proceeds.*

> **Definition:** In LangGraph, **state is a shared memory that flows through your workflow**. It holds all the data being passed between nodes as your graph runs.

When you build a graph, LangGraph asks you to **first define your state**, adding all data points as **key-value pairs**. *Data points in the UPSC workflow: essay text, essay topic, depth score, language score, clarity score, overall score.*

**The most powerful property of state — every node has access to it:**

1. When a node executes, it receives the **entire state as input**.
2. The node performs its work and **makes changes to the state**.
3. It passes the **updated state** to the next node.

So state is **(a) shared** between all nodes and **(b) mutable** — all nodes can change it — and it **evolves** as execution moves forward.

**How is state created in code?** It's a special dictionary called a **TypedDict** (a class in Python). You create an object of this class and add all the fields. You can also use a **Pydantic object**, but mostly a TypedDict is used.

---

## 5. Reducers

> Reducers are **closely connected to the State concept.**

Recall the two key properties of state: it is **accessible to all nodes**, and it is **mutable** (any node can change it).

**The problem with plain updating:**

**Scenario A (works fine):** A basic workflow takes two numbers → computes their sum → multiplies the result by 2 → prints it. State data points: `first_number`, `second_number`, `result`. The sum node writes `result = 11`; the next node updates it to `result = 22`. **Updating** means *removing the old value and putting a new value* — and here that's exactly what we want. Multiple nodes update the same value, last write wins.

**Scenario B (updating fails) — a simple chatbot:** A human node and an LLM node talk **in a loop**. State has one key: `messages`.

- Human: *"Hi, my name is Nitesh"* → stored in `messages`.
- LLM node sees it → replies *"Hi, how can I help you?"* → this **replaces** the previous message.
- Human: *"Can you tell me my name?"* → this **replaces** again.
- Now the LLM **can never** tell the user's name, because the message where the name was given got **erased** from state.

> In a chatbot, the **update (replace) policy fails**. Ideally you should **add** every new message to the existing ones, not replace.

**This is what a reducer does.** A reducer tells LangGraph **how updates to the state should be applied** — whether to **replace**, **add (append)**, or **merge**.

> **Definition:** Reducers in LangGraph define **how updates from nodes are applied to the shared state**. **Each key in the state can have its own reducer**, which determines whether new data **replaces, merges, or adds to** the existing value.

**Another "add" example — the UPSC workflow:** The `essay_text` key stores the student's essay. If the first essay scores poorly and the student rewrites, the variable's value changes and the **old essay is lost**. But if the student wants to see their **evolution** — first, second, and third essays — to track how they improved, the **update policy is not good**. You instead use an **add** function so the previous essay is **not erased**; new ones are appended.

> Reducers are especially useful in **parallel workflows** — when multiple nodes write to the same key simultaneously, a reducer defines how those writes combine.

---

## 6. LangGraph's Execution Model

> A conceptual look at how LangGraph executes a workflow **behind the scenes**.

**Interesting fact:** LangGraph's execution model is **inspired by Google Pregel** — a system that can do **large-scale graph processing**, integrated into many Google products.

**The phases — what happens when you build and run a workflow:**

**A. Graph Definition** — You create a graph by defining three things: its **nodes**, its **edges**, and its **state** (a TypedDict).

**B. Compilation** — You call a **`compile`** function. Its main purpose is to check that the graph's structure is **logically correct** — e.g., no **orphan node** (a node not connected to any other node), and no other structural inconsistencies.

**C. Execution Phase:**

1. **Invocation** — You pass an **initial state** to the graph's **first node**.
2. The first node **activates** — meaning its attached **Python function is called** — does its work, and makes a **partial update** to the state.
3. The **updated state** travels through an **edge** to the next node, which activates the same way.
4. This continues node by node until the end.

> **Message passing:** The process of using **edges to pass the state to the next node** is called message passing.

**Superstep — why "super"?** The round-by-round work is called a **superstep** in LangGraph's terminology, *not* just a "step." The reason: when the graph has **parallel nodes**, one round may consist of **multiple steps executing in parallel**. *If a node's update flows to three parallel nodes, the system sends the message to all three, they all work simultaneously and all update the state.* Since more than one step can run in parallel, calling it a plain "step" isn't logical — hence **superstep**.

> A workflow is **divided into supersteps**. **One superstep can have one step, or it may have more than one step** (parallel).

After parallel nodes finish, the state updates are **merged via reducers**, then passed on through edges.

**When does execution stop?** The whole workflow stops when **(1) there is no active node** and **(2) no message is being passed through any edge**. When both conditions are true, the workflow halts.

> **Key takeaway:** You do **not** manually call one node after another. You call the first node, give it state, and all the rest happens **internally** — via **message passing** (sending state through edges) and **supersteps** (the workflow divided into rounds, each possibly containing parallel steps).

---

## 7. Quick Revision Summary

**LangGraph** = an **orchestration framework** that represents any **LLM workflow as a graph** (nodes = tasks, edges = order), and also provides parallel execution, loops, branching, memory, and resumability.

**LLM workflows** = step-by-step processes for building complex LLM apps, where tasks involve prompting, reasoning, tool calling, memory access, or decision making. **Five common workflows:**

| Workflow | Core idea |
|----------|-----------|
| **Prompt Chaining** | Call the LLM multiple times in sequence; break a complex task into sub-tasks |
| **Routing** | An LLM acts as a router, deciding which downstream LLM handles a query |
| **Parallelization** | Split into **predefined** sub-tasks, run in parallel, aggregate results |
| **Orchestrator-Worker** | Like parallelization, but sub-tasks are **decided dynamically** by an orchestrator |
| **Evaluator-Optimizer** | Generator + Evaluator LLMs loop with feedback until the output is accepted |

**Graphs, Nodes, Edges** — the most important core concept. **Nodes** = tasks (each is a **Python function**); **Edges** = which node runs next. *Nodes say what to do; edges say when.* Edge types: sequential, parallel, conditional, loop.

**State** = a **shared, mutable memory** flowing through the workflow as key-value pairs; **every node** reads it as input and writes updates to it; it **evolves** over execution. Created in code as a **TypedDict** (or Pydantic object).

**Reducers** = define **how state updates are applied** — **replace**, **add (append)**, or **merge**. Each key can have its own reducer. Updating (replace) is fine for a sum workflow but fails for a chatbot, where messages must be **appended**. Especially useful in parallel workflows.

**Execution Model** (inspired by **Google Pregel**): **Graph Definition** (nodes + edges + state) → **Compilation** (`compile` checks structure, e.g., no orphan nodes) → **Execution** (invocation passes initial state to the first node; nodes **activate**, make **partial updates**, and pass state via **message passing**). The run is divided into **supersteps** (each can contain one or more parallel steps). Execution **stops** when no node is active and no message is in transit.

---

*Revision notes — LangGraph Core Concepts (Playlist Video 4).*
