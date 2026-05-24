# LangChain vs LangGraph — Why LangGraph Exists (Revision Notes)

---

## Table of Contents

1. [Frameworks for Building Agentic AI](#1-frameworks-for-building-agentic-ai)
2. [Quick Recap — What is LangChain?](#2-quick-recap--what-is-langchain)
3. [Workflow vs Agent — A Critical Distinction](#3-workflow-vs-agent--a-critical-distinction)
4. [The Automated Hiring Workflow (Running Example)](#4-the-automated-hiring-workflow-running-example)
5. [The 8 Challenges of Building Complex Workflows in LangChain](#5-the-8-challenges-of-building-complex-workflows-in-langchain)
   - [Challenge 1 — Control Flow Complexity](#challenge-1--control-flow-complexity)
   - [Challenge 2 — State Handling](#challenge-2--state-handling)
   - [Challenge 3 — Event-Driven Execution](#challenge-3--event-driven-execution)
   - [Challenge 4 — Fault Tolerance](#challenge-4--fault-tolerance)
   - [Challenge 5 — Human-in-the-Loop](#challenge-5--human-in-the-loop)
   - [Feature 6 — Nested Workflows (Subgraphs)](#feature-6--nested-workflows-subgraphs)
   - [Challenge 7 — Observability](#challenge-7--observability)
6. [What is LangGraph?](#6-what-is-langgraph)
7. [When to Use What?](#7-when-to-use-what)
8. [Should You Stop Learning LangChain?](#8-should-you-stop-learning-langchain)
9. [Quick Revision Summary](#9-quick-revision-summary)

---

## 1. Frameworks for Building Agentic AI

Building Agentic AI applications is **genuinely difficult** — you cannot just write the whole thing from scratch in Python. This is why several **frameworks** exist to make it much easier, such as **CrewAI**, **Microsoft AutoGen**, and others.

The framework used throughout this playlist is **LangGraph**, built by the **LangChain team**. Since LangChain is already a very popular product and the feedback on LangGraph has been strong, LangGraph is considered **one of the top frameworks** for building Agentic AI.

> **Prerequisite:** You should already have a basic idea of **LangChain** — what it is, what it can do, and how basic code is written in it. If not, watch the first two videos of the LangChain playlist (*Introduction to LangChain* and *LangChain Components*) first.

---

## 2. Quick Recap — What is LangChain?

**Definition:** LangChain is an **open-source library** designed to simplify the process of building **LLM-based applications**.

After LLMs arrived, a culture emerged of **integrating LLMs into every kind of software** — adding a chatbot to a SaaS product, a Chrome plugin to chat with YouTube videos, etc. These are called **LLM-based applications**, and they are hard to build because many pieces must be joined together. LangChain simplifies this whole process.

**LangChain's modular building blocks (components):**

| Component | What it does |
|-----------|--------------|
| **Models** | A **unified interface** to talk to any LLM provider (OpenAI, Anthropic's Claude, Hugging Face models, Ollama). Swapping one LLM for another needs almost no code change. |
| **Prompts** | Helps you design and engineer prompts in many ways — LLMs work entirely on prompts. |
| **Retrievers** | Fetches relevant documents from a vector store / knowledge base using various strategies and algorithms. Used to build **RAG-based** applications. |
| **Chains** | LangChain's **biggest offering** — connect components together so the output of one block automatically becomes the input of the next. |

**The Chain concept:** You join components (e.g., Prompt → Model → Output Parser) into a chain. The output of the first block automatically feeds the second, and so on — **you do not have to wire this manually**.

**What you can build with LangChain:**

- **Simple conversational workflows** — chatbots, text summarizers (take a prompt → send to LLM → show output, optionally in a loop).
- **Multi-step workflows** — e.g., take a topic → generate a detailed report → generate a summary from that report (a longer chain).
- **RAG-based applications** — user prompt → retriever fetches relevant *context* from a vector store → context + prompt sent to LLM → grounded response.
- **Basic-level agents** — via the **Tools** concept: connect an LLM to tools (APIs or Python functions); the LLM decides *when* to call *which* tool with *what* input. *Example:* give the LLM a weather API tool — when the user asks "what's the weather in Gurgaon?", the LLM triggers the tool and formats the result.

---

## 3. Workflow vs Agent — A Critical Distinction

> This distinction is critical. The hiring flowchart shown in this video is **NOT an Agentic AI application — it is a workflow.**

Based on Anthropic's blog *"Building Effective Agents"*:

- **Workflows** are systems where LLMs and tools are orchestrated through **predefined code paths**.
- **Agents** are systems where LLMs **dynamically direct their own processes and tool usage**, maintaining control over how they accomplish tasks.

| | Workflow | Agent |
|---|----------|-------|
| Who designs the flowchart? | The **developer**, in advance | The **agent itself**, at runtime |
| Is the flow static or dynamic? | **Static** — runs the same way every time | **Dynamic** — different flow on each run |
| Autonomy | Low | High |
| Uses LLMs? | Yes | Yes |

In the last video's hiring example, the **agent** decided the steps and their order. In *this* video, the flowchart is **pre-built by the developer** and executes in the **same order every time** — that is why it is a **workflow**, not an agent.

> **Plan for this video:** First build the hiring *workflow* as a flowchart, then conceptually try to build it in **LangChain** and see what challenges arise.

---

## 4. The Automated Hiring Workflow (Running Example)

A complete, detailed flowchart for the automated hiring scenario from the previous video. Step by step:

1. **Receive hiring request** — via a prompt: *"We need to hire a remote Back-End Engineer with 2–4 years experience."*
2. **Create JD** — an LLM creates a detailed Job Description.
3. **JD approval (human supervisor)** — if not approved, loop back to *Create JD* with feedback; if approved, continue.
4. **Post JD** — using tools like the **LinkedIn API** and **Naukri.com API**.
5. **Wait 7 days** — to let applications accumulate.
6. **Monitor applications** — check the application count via the LinkedIn API. Suppose the **threshold is 20 applications**.
7. **Conditional check — enough applications?**
   - **No** → **Modify the JD** (e.g., lower eligibility criteria, include freshers, change "Back-End" to "Full-Stack", raise salary) → **wait 48 hours** → monitor again. *(This forms a loop until the exit condition is met.)*
   - **Yes** → proceed to shortlisting.
8. **Shortlisting** — a **résumé parser tool** downloads and parses all résumés; an LLM scores each one. Candidates above a score threshold are shortlisted.
9. **Schedule interviews** — using the **Calendar API** and **Mail API**: check interviewer availability and email the candidates.
10. **Conduct interviews** — provide a question bank, send reminder mails, run the interviews.
11. **Selected?** — for each candidate:
    - **No** → send a regret email.
    - **Yes** → send an **offer letter** (LLM creates it, Mail API sends it).
12. **Offer accepted?** — track the response.
    - **Not accepted** → **renegotiate** (a human does this), send a revised offer, wait again.
    - **Accepted** → start **onboarding**.
13. **Onboarding** — via integration with the **HR Management System**: send welcome email, plan a KT session, provision a laptop, etc. The hiring process ends.

> Conceptually clear. The big question: can this whole process be **implemented in code using LangChain** — and what challenges appear?

---

## 5. The 8 Challenges of Building Complex Workflows in LangChain

LangChain handles **basic, linear workflows** easily — especially **linear chains**. But the hiring flowchart above is a **complex, non-linear workflow**. The video walks through challenges that show *why* this is hard in LangChain and *how* LangGraph solves each.

---

### Challenge 1 — Control Flow Complexity

LangChain is mostly used to build **chains**, and a chain is a **linear workflow**. The hiring flowchart, however, is **highly non-linear**. Three things cause this non-linearity:

| Source of non-linearity | Explanation |
|-------------------------|-------------|
| **Conditional branches** | Based on a condition, control flow goes one direction or another. *E.g., enough applications → one path; not enough → another.* |
| **Loops** | Control repeats a section. *E.g., keep regenerating the JD until it is approved.* |
| **Jumps** | Control suddenly moves forward or back. *E.g., after waiting 48 hours, jump back earlier in the logic.* |

**Why LangChain struggles:** LangChain has **no built-in constructs** for conditional branching, loops, or jumps. You must implement these yourself in plain **Python** (e.g., a `while` loop, `if/else`). Any code you write *outside* the library just to stitch the flow together is called **glue code**.

> **The less glue code, the better.** As a complex application grows, glue code piles up — and more glue code means **harder to maintain, harder to debug, harder for teams to work on**. *This is the biggest flaw of LangChain: it works great for linear chains, but raises its hands the moment non-linearity enters.*

**How LangGraph solves it:** In LangGraph you represent the **entire workflow as a graph** (hence the name). Each task is a **node** (a plain Python function), and **edges** connect nodes and decide control flow. Since a **graph is a non-linear data structure**, any complexity is represented easily.

- **Conditional edges** let you route based on a condition (e.g., *check approval* node → post JD if yes, loop back to create JD if no).
- Loops, branching, and jumps all come **built-in** — **zero glue code**. You don't write `while` or `if/else` yourself; LangGraph handles it.

> Result: **high maintainability** for arbitrarily complex applications.

---

### Challenge 2 — State Handling

**What is state?** A complex workflow has several important **data points** whose values matter, e.g.:

- The **JD** itself
- Whether the JD is **approved**
- Whether the JD is **posted**
- How many people have **applied**
- The **minimum applications** needed to start interviews
- How many candidates were **shortlisted** (and their contact details)
- How many **offers** were sent, the **offer status**, the **onboarding status**

These data points and their values **evolve over time** as the workflow progresses. *Example:* at *Hiring Request* the JD value is `None`; at *Create JD* it gets set; at *JD Approved* the status becomes true/false; at *Post JD* the posted flag becomes true.

> **The complete set of these data points is called the STATE of the workflow.** No workflow can execute correctly unless its state is tracked properly.

**Why LangChain struggles:** State is a set of **key-value pairs**, but LangChain gives **no option to store and track such key-value pairs**. LangChain has a memory concept, but it is **conversational memory** (storing your chat with the LLM) — not workflow state. So workflows built in LangChain are **stateless**. To add state, you must do it **manually**: create a global dictionary at the top of your code and manually update its values at every step of a long chain — **hectic and error-prone**.

> Your only two options in LangChain: treat the entire state as conversational memory (text), or manually maintain a dictionary and pass it around — both painful for complex workflows.

**How LangGraph solves it:** LangGraph execution is **stateful**. When you create the graph you define a **state object** (using **Pydantic** or a **TypedDict** — essentially a dictionary). This state object is:

- **Accessible** by **every node** of the graph (each node can *read* it).
- **Mutable** — every node can also *edit* it.

Each node receives the **state as input** and returns the **state as output**. As execution moves node by node, the updated state flows to the next node, so information passing is seamless.

> **LangChain is stateless; LangGraph is stateful.** For complex workflows, LangGraph is far better suited.

---

### Challenge 3 — Event-Driven Execution

Any workflow can execute in two ways:

- **Sequential** — runs left-to-right **without stopping**; as one block finishes, the next starts immediately.
- **Event-driven** — the workflow **pauses** somewhere and waits for an **external trigger**; when the trigger arrives, it **resumes**.

The hiring workflow has **multiple event-driven points**:

- After **posting the JD**, *Monitor Applications* runs only after a **7-day wait** completes.
- After **modifying the JD**, the workflow pauses, then resumes after a **2-day wait**.
- After **sending the offer letter**, the next step runs only when the candidate **accepts or rejects** (the external trigger is the candidate's reply).

**Why LangChain struggles:** LangChain is **not built for event-driven execution** — it was designed for **sequential execution** (once a chain starts, it finishes its work and stops). To implement this you must **split the workflow into two chains**, write external Python code to track elapsed time, trigger the second chain, and manually code the **state transfer** between them — again, lots of **glue code**.

**How LangGraph solves it:** LangGraph **inherently provides** event-driven execution. Because execution is **stateful**, at any node you can **save the current state** using a feature called a **checkpointer** (in memory or in an external database). You pause, wait for the external trigger, then resume **exactly from the saved state**.

---

### Challenge 4 — Fault Tolerance

**Fault tolerance** means: if something goes wrong in a system, can it **recover** and continue running properly? It is especially important for **long-running** workflows.

The hiring workflow **is long-running** — create JD, post, wait 7 days, modify and wait 2 days, schedule, interview, offer, onboard — it can run for **days or even months**. Longer runs → higher chance of faults.

**Two types of faults:**

- **Small fault** — a fault at the **node level**. *E.g., the LinkedIn API is down while posting the JD.*
- **Big fault** — a **system-level** fault. *E.g., the AWS server (or Docker container) running the workflow goes down entirely.*

**Why LangChain struggles:** LangChain has **no fault-tolerance concept**. If a 5-step chain fails at step 3, you must **restart the chain from the beginning**. LangChain assumes its chains are **short-lived** (trigger → finish quickly), so fault tolerance was not considered important.

**How LangGraph solves it:** LangGraph provides **built-in fault tolerance** for both fault types:

- **Small faults → Retry logic.** Wrap a node so that if an error occurs (e.g., LinkedIn API down), it is caught and retried after a short delay.
- **Big faults → Recovery.** Because execution is stateful, LangGraph saves a **checkpoint** (a state snapshot) after **every node's execution**, in memory or an external **persistence layer**. If the system crashes, a **resume** function automatically identifies the previous state, figures out which node failed and what the next node is, and **restarts execution from exactly that point** — not from the beginning.

> Retry **and** recovery logic are both built into LangGraph.

---

### Challenge 5 — Human-in-the-Loop

**Human-in-the-loop (HITL):** at some stage in a workflow, you need a **human's decision**. *Example:* in the hiring workflow, right after the JD is created it must be **approved by a human**; the workflow cannot proceed without that approval. Another example: asking a human before **posting** the JD to a website.

> In real-world workflows there are many points where you want **control with the human**, not the agent — because **accountability for risky actions should rest with a human**. This pause for a human is called human-in-the-loop.

**Why LangChain struggles:** LangChain has **no default mechanism** to pause a chain, wait for a human, and resume after approval. A chain is **synchronous and sequential**, so asking for input is only fine for **short durations**. If approval takes, say, **24 hours**, the script keeps running in the same state the whole time — **consuming compute resources** and risking a mid-wait crash. The workaround is again **splitting the chain into two parts** and manually passing the state — lots of glue code and maintainability problems.

**How LangGraph solves it:** In LangGraph, human-in-the-loop is a **first-class citizen** — explicitly added when the framework was built; the documentation even has a dedicated section. Per the docs, LangGraph lets you **pause execution indefinitely** — for minutes, hours, or even days — until human input is received. This works because LangGraph **checkpoints the graph state after every step**, allowing the system to **persist execution context** and later **resume from where it left off**, supporting **asynchronous human review without time constraints**.

> **Analogy:** It is like playing a video game — you can **save your progress** at Stage 3, close the game, and **resume at Stage 3** tomorrow. LangGraph gives the workflow exactly this mechanism.

> Challenges **3, 4, and 5 are all connected** — they all rely on **stateful execution** and the **checkpointer**.

---

### Feature 6 — Nested Workflows (Subgraphs)

> This is called a **feature**, not a challenge.

**Nested workflows** mean building a **workflow inside another workflow**. In LangGraph, any workflow is a **graph** of nodes. It is possible to **replace a single node with an entire other graph** — i.e., a node can itself *be* a graph. These are called **subgraphs**.

> **Definition (from the docs):** A **subgraph** is a graph that is used as a **node** in another graph. This is the concept of **encapsulation** applied to LangGraph — subgraphs let you build complex systems whose components are themselves graphs.

**Why it's useful — the hiring example:** The *Conduct Interview* step looks like one node, but it is actually very complex: generate per-candidate questions, run multiple interview rounds (Round 1 → evaluate → Round 2 → evaluate → Round 3 → evaluate). So *Conduct Interview* can be treated as a **separate workflow** connected into the larger one.

**Two big use cases of subgraphs:**

1. **Multi-Agent Systems** — a system where multiple agents work together. *Example: a self-driving car* — one agent processes sensor data, another handles driving capability, another handles in-car entertainment, and a fourth acts like a **CEO** coordinating the rest. Multi-agent systems are widely deployed to solve complex problems.
2. **Reusability** — make a graph **reusable** and use it as-is in many places of a larger graph. *Example:* build one small **approval** subgraph and reuse it everywhere approval is needed (approving the JD, posting the JD, scheduling interviews). This is just like writing **reusable functions** in programming.

**LangChain:** This feature is **not available** — you cannot nest workflows. For LangChain it is a challenge; for LangGraph it is a **feature you have**.

---

### Challenge 7 — Observability

**Definition:** Observability refers to how easily you can **monitor, debug, and understand what your workflow is doing at runtime**.

At runtime many problems can occur — errors, crashes, or unexpected decisions. Especially **in production** (deployed, with real users), it is vital to **monitor closely** how the agent/workflow runs. This helps with both **debugging** and **auditing**. *Example:* if an agent ran unlimited paid ads on LinkedIn and overspent, you need to trace back *why* and *what steps* led to that.

**LangChain — partial observability:** Observability **is available** in LangChain via a library called **LangSmith**, whose purpose is to monitor LLM-based applications. Once integrated, LangSmith records, per step: that an LLM was called, what prompt was sent, what reply came back, how many tokens it used, how long it took.

> **The problem:** LangSmith can monitor **only the LangChain code** — it **cannot monitor your glue code**. It won't understand what is happening inside *your* `while` loop, or which loop iteration an LLM call belongs to. So complex LangChain apps get only **partial observability**, not complete.

**How LangGraph solves it:** LangGraph has a **tight integration** with LangSmith. Since LangGraph execution is fully **stateful** and there is **no glue code**, everything is tracked — which node executed when, how the state changed before vs. after each node, the messages exchanged between human and agent, and at which point a human gave approval. A **complete chronological timeline** of the run is recorded, which you can fully **back-track** using LangSmith.

> Complex app in LangChain → **partial observability**. Complex app in LangGraph → **complete observability**.

---

## 6. What is LangGraph?

> **Definition:** LangGraph is an **orchestration framework** that enables you to build **stateful, multi-step, and event-driven workflows** using LLMs. It is **ideal for designing both single-agent and multi-agent Agentic AI applications**.

Think of LangGraph as a **flowchart engine for LLMs**: you define the steps as **nodes**, how they are connected using **edges**, and the logic governing the transitions. LangGraph takes care of **state management, conditional branching, looping, pausing and resuming, and fault recovery** — features essential for building robust, production-grade AI systems.

---

## 7. When to Use What?

| Use **LangChain** when… | Use **LangGraph** when… |
|--------------------------|--------------------------|
| Building **simple, linear workflows** | Building **complex, non-linear workflows** |
| Prompt chains, summarizers, basic RAG systems | You need conditional branches, loops, jumps |
| Short-running tasks | You need **human-in-the-loop** steps |
| | You need **multi-agent** coordination/collaboration |
| | You need **asynchronous, event-driven** execution |

> When you pick up a new project, understanding its requirements should immediately tell you whether to use LangChain or LangGraph.

---

## 8. Should You Stop Learning LangChain?

**No.** You still need LangChain.

- **LangGraph is built on top of LangChain.** It was **not** designed to *replace* LangChain — it was designed to solve **more complex problems**, and it solves them **with the help of LangChain**.
- Even in complex workflows you still need to interact with LLMs, write prompts, and load documents — and those components (**ChatOpenAI, prompt templates, retrievers, document loaders, text splitters, tools**) still come from **LangChain**.
- **LangChain's purpose:** provide the **components** (plus a basic workflow system — chains).
- **LangGraph's purpose:** **orchestrate** a workflow / framework — *not* to provide components.

> The two work **hand in hand**. Throughout the playlist, both LangChain and LangGraph are used together. The effort you spent learning LangChain is **not wasted**.

---

## 9. Quick Revision Summary

**Frameworks** like CrewAI, AutoGen, and **LangGraph** exist because building Agentic AI from scratch is hard. This playlist uses **LangGraph** (by the LangChain team).

**LangChain** = an open-source library to build LLM-based apps, with modular components — **Models, Prompts, Retrievers, Chains**. Great for **linear** workflows.

**Workflow vs Agent:** A **workflow** runs on **predefined developer code paths** (static). An **agent** **dynamically directs its own process** (dynamic). The hiring flowchart is a *workflow*.

**The 8 challenges of building complex workflows in LangChain — and how LangGraph fixes them:**

| # | Challenge | LangChain | LangGraph |
|---|-----------|-----------|-----------|
| 1 | **Control flow complexity** | No constructs for branches/loops/jumps → glue code | Workflow as a **graph** of nodes + edges; loops/branching built-in, zero glue code |
| 2 | **State handling** | Stateless; manual global dictionary | **Stateful**; shared, mutable **state object** every node reads/writes |
| 3 | **Event-driven execution** | Built only for sequential runs | Pause/resume via **checkpointer**; built-in |
| 4 | **Fault tolerance** | None; restart from the beginning | **Retry** (small faults) + **recovery** (big faults) via checkpoints |
| 5 | **Human-in-the-loop** | No pause/wait mechanism; bad for long waits | **First-class citizen**; pause indefinitely until human input |
| 6 | **Nested workflows** | Not possible | **Subgraphs** → multi-agent systems + reusability |
| 7 | **Observability** | Partial (LangSmith can't see glue code) | **Complete** observability via tight LangSmith integration |

*(Challenges 3, 4, and 5 all rely on **stateful execution** + the **checkpointer**.)*

**LangGraph** = an **orchestration framework** for **stateful, multi-step, event-driven** workflows with LLMs — a **flowchart engine for LLMs** (nodes, edges, transition logic).

**When to use what:** LangChain → simple linear workflows. LangGraph → complex non-linear workflows (conditionals, loops, HITL, multi-agent, event-driven).

**Don't drop LangChain:** LangGraph is **built on top of** LangChain and uses its components. LangChain provides the **components**; LangGraph provides the **orchestration**. They work **hand in hand**.

---

*Revision notes — LangChain vs LangGraph (Playlist Video 3).*
