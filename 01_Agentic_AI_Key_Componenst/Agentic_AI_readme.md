# Agentic AI — Introduction 



---

## Table of Contents

1. [What is Agentic AI?](#1-what-is-agentic-ai)
2. [Generative AI vs Agentic AI — A Practical Example](#2-generative-ai-vs-agentic-ai--a-practical-example)
3. [The 6 Key Characteristics of an Agentic AI System](#3-the-6-key-characteristics-of-an-agentic-ai-system)
   - [3.1 Autonomy](#31-autonomy)
   - [3.2 Goal-Oriented](#32-goal-oriented)
   - [3.3 Planning](#33-planning)
   - [3.4 Reasoning](#34-reasoning)
   - [3.5 Adaptability](#35-adaptability)
   - [3.6 Context Awareness](#36-context-awareness)
4. [The 5 Core Components of an Agentic AI System](#4-the-5-core-components-of-an-agentic-ai-system)
5. [Quick Revision Summary](#5-quick-revision-summary)

---

## 1. What is Agentic AI?

**Definition:** Agentic AI is a type of AI that can take up a task or goal from a user and then work towards completing it on its own, with minimal human guidance. It plans, takes actions, adapts to changes, and seeks help only when necessary.

In simple terms, Agentic AI is a **software paradigm** where:

- You provide the system with a **goal**.
- The system then **thinks for itself** about how that goal can be achieved.
- All the **planning** and **execution** required to achieve the goal is done by the system itself.
- **Human involvement is kept to a minimum.**

This is fundamentally different from other software paradigms. For example, a **Generative AI chatbot** is **reactive** — it only answers the exact question you ask at that moment and cannot take initiative on its own. An Agentic AI system, by contrast, is **proactive and autonomous**.

| Aspect | Generative AI (Chatbot) | Agentic AI |
|--------|-------------------------|------------|
| Behavior | Reactive | Proactive |
| Initiative | None — answers only what is asked | Takes its own initiative |
| Human role | Drives every step | Sets the goal; minimal supervision |
| Scope of answer | "To the point" — no more, no less | Plans and executes the whole task |

---

## 2. Generative AI vs Agentic AI — A Practical Example

### Example A: Planning a trip to Goa 🏖️

Travelling involves many decisions: figuring out dates, choosing a mode of transport, booking hotels, deciding where to visit and what to eat.

**With a Generative AI chatbot (reactive):**
At *every step* you must ask a separate question, and the bot answers only that question.
- *"What's the best way to reach Goa on the 15th?"* → It says: "You can take a flight."
- *"Which hotels are best for this duration?"* → It gives hotel recommendations.
- *"Where can I visit in Goa based on the weather?"* → It answers only that.

You are doing all the orchestration. The process is **reactive** — you ask, it replies, to the point.

**With an Agentic AI system (proactive):**
You simply say: *"I want to travel to Goa between these dates."*
Everything after that is handled by the system itself — it finds the best travel option, gives hotel recommendations, and plans your full itinerary on its own.

> **Main takeaway:** Agentic AI is *completely autonomous* and acts *proactively*, whereas a reactive paradigm waits for instructions at every step.

---

### Example B: The AI HR Recruiter 🧑‍💼 (used throughout the notes)

**Scenario:** You are an **HR recruiter**. Your task is to **hire a Back-End Engineer** for your company. Your company has given you an **Agentic AI chatbot** to help.

**Step 1 — Provide the goal**
You tell the chatbot: *"I want to hire a Back-End Engineer."* You add detail — remote hiring, 2–4 years of experience, etc.

**Step 2 — The agent understands the goal and builds a plan**
A proposed plan might be:
1. Draft a Job Description (JD).
2. Post the JD on the best job platforms.
3. Continuously monitor how many people apply; tweak if applications are low.
4. Once enough applications arrive, screen candidates.
5. Schedule interviews for screened candidates.
6. Send an offer letter to the selected candidate.
7. Start the onboarding process for those who accept.

**Step 3 — Execution (autonomously, step by step)**

- **Draft the JD:** The agent accesses company documents to understand the required technologies, the salary range for 2–4 years of experience, and job responsibilities. It then shows you the draft for approval.
- **Post the JD:** Using tool access (e.g., the **LinkedIn API** and the **Naukri API**), it posts the job and notifies you.
- **Monitor & adapt:** After a few days only 2 people have applied — fewer than expected. The agent proactively suggests changes: (1) change "Back-End Engineer" to "Full-Stack Engineer" in the JD, (2) run a paid ad on LinkedIn. It **asks your permission** before acting.
- **Screen candidates:** Once enough applications arrive, it uses a **résumé parser tool** to download and analyze résumés, then reports: e.g., "2 strong candidates, 3 partial matches, 3 weak matches."
- **Schedule interviews:** It checks **your calendar**, finds you are free Friday, drafts an email, and sends it to you and the candidates.
- **Reminders & support:** On interview day it reminds you and sends a list of suggested interview questions in a document.
- **Offer letter:** When you select a candidate, it drafts an offer letter from company documents, lets you review it, and sends it via your email.
- **Onboarding:** When the candidate accepts, it sends a welcome email, submits an IT access request, provisions a laptop, and offers to set up a joining-day meeting.

> **The most striking feature:** how **autonomous** the agent is. You gave it just one goal — it planned, executed, and adapted on its own, asking for permission only when needed.

---

## 3. The 6 Key Characteristics of an Agentic AI System

If someone shows you a chatbot or AI application and asks "Is this an Agentic AI system?", check it against these **six traits**. If the answer is **yes to all six**, it is an Agentic AI system.

1. Autonomy
2. Goal-Oriented
3. Planning
4. Reasoning
5. Adaptability
6. Context Awareness

---

### 3.1 Autonomy

**Definition:** Autonomy refers to an AI system's ability to make decisions and take actions on its own to achieve a given goal, **without needing step-by-step human instructions**.

Autonomy implies the system is **proactive** — it does things before being told. *Example:* the HR agent realized on its own, after 3 days, that too few people had applied and that changes were needed. A reactive tool like ChatGPT would require you to monitor the job posting yourself and then ask what to do.

**Autonomy shows up in multiple aspects of the system:**

- **Execution autonomy:** The agent automatically executes each step of the plan one by one.
- **Decision-making autonomy:** It decides *how many* candidates to shortlist and *on what basis* — with minimal human intervention.
- **Tool-usage autonomy:** It decides *which* tool to use and *when* (mail tool, calendar tool, résumé parser, etc.).

> Autonomy is not limited to one aspect — it appears in **every aspect** of how an Agentic AI system works.

**Controlling Autonomy** — Autonomy is powerful but **risky**, so it must be controlled. Ways to control it:

| Control Mechanism | What it does |
|-------------------|--------------|
| **Scope of permissions** | Limit which tools and actions the agent can perform independently. *E.g., "screen all résumés freely, but ask me before rejecting anyone."* |
| **Human-in-the-loop** | Insert checkpoints where human approval is required before continuing. *E.g., the agent drafts a JD but must ask before posting it.* |
| **Override controls** | Allow users to stop, pause, or change the agent's behavior at any time. *E.g., a "pause hiring" command.* |
| **Guardrails & policies** | Define hard rules and ethical boundaries the agent must follow. *E.g., "never schedule interviews on weekends" or "never use informal language in emails."* |

**Why control matters — dangers of uncontrolled autonomy:**
- Rolling out job offers with **incorrect salaries / terms** without asking.
- **Biased shortlisting** (e.g., by nationality or age) — illegal in many countries.
- Spending **unlimited money** on paid ads without permission.

---

### 3.2 Goal-Oriented

**Definition:** Being goal-oriented means the AI system operates with a **persistent objective** in mind and continuously directs its actions to achieve that objective, rather than just responding to isolated prompts.

Everything the AI Recruiter did was for **one single goal**: hire a Back-End Engineer.

> **Autonomy and Goal-Orientation move hand in hand.** Autonomy lets the system function independently, but the **goal tells it *what* to do**. The goal acts like a **compass** for the system's autonomy — without a goal, the system cannot function autonomously in a meaningful direction.

**Two ways to define goals:**

- **Independent goal:** *"Hire a Back-End Engineer."*
- **Goal with constraints:** *"Hire a Back-End Engineer **from India**"*, or *"only **remote** hiring"*, or *"spend only **$X**."* Here "from India", "remote", and the budget are **constraints**.

**How goals are stored:** Goals are stored in the agent's **core memory**. A rough conceptual representation (the exact format varies by library):

```json
{
  "main_goal": "Hire a Back-End Engineer",
  "constraints": {
    "experience_years": "2-4",
    "remote": true,
    "tech_stack": ["..."]
  },
  "status": "active",
  "created_at": "2025-01-01",
  "progress": {
    "jd_created": true,
    "posted_on": ["LinkedIn", "Naukri"],
    "applications_received": 8,
    "interviews_scheduled": 2,
    "onboarding": false
  }
}
```

When hiring completes, `status` becomes `completed`.

**Goals can be altered mid-way.** *Example:* you were hiring a Back-End Engineer; after 7 days nobody applied, so you decide instead to *"find a freelancer."* The main goal changes → planning changes → the agent executes a different plan.

---

### 3.3 Planning

> Probably the **most important** characteristic. Agentic AI systems essentially operate in **two steps**: **(1) Planning** and **(2) Execution**.

This two-step process is **iterative (runs in a loop)**. If, mid-execution, the agent realizes step 4 is impossible, it returns to the planning stage, re-plans, and executes the new plan. This repeats until the goal is achieved.

**Definition:** Planning is the agent's ability to break down a high-level goal into a **structured sequence of actions and sub-goals**, then decide the best path to achieve the desired outcome.

**Planning is a search problem** — you have an **initial state** (company needs a Back-End Engineer) and a **final state** (Back-End Engineer hired). Multiple paths exist between them; the agent must choose the most efficient / optimized one.

**Planning is done in 3 steps:**

**Step 1 — Generate multiple candidate plans**
The agent does *not* create a single plan; it creates **multiple plans** ("candidate plans").
- *Plan A:* Draft a JD and post + promote it on LinkedIn, GitHub Jobs, and AngelList.
- *Plan B:* Instead of job portals, run an internal **referral** program or approach a **hiring agency**.

**Step 2 — Evaluate the plans**
Decide which plan is best, using criteria such as:

| Criterion | Question it answers |
|-----------|---------------------|
| **Efficiency** | Which plan is faster to execute? |
| **Tool availability** | Does the agent have the tools a plan needs? *(If Plan A needs a Google Search API the agent doesn't have, that plan is rejected.)* |
| **Cost** | Which plan fits the budget? *(Referrals/agencies cost more; job portals cost less.)* |
| **Risk** | Which plan has a higher risk of failure? |
| **Alignment with constraints** | *E.g., for remote hiring — will LinkedIn or internal referrals give more remote candidates?* |

**Step 3 — Select the best plan**
Based on the above metrics, the best plan is chosen. Selection can happen via:
- **Human-in-the-loop input** — the agent asks the human supervisor to pick.
- **A pre-programmed policy** — the agent decides based on built-in rules.

> **In a nutshell:** Planning = create a structured sequence of sub-goals → *generate* multiple plans → *evaluate* them → *select* one → execute it.

---

### 3.4 Reasoning

**Definition:** Reasoning is the cognitive process through which an Agentic AI system **interprets information, draws conclusions, and makes decisions** — both while *planning* and while *executing*.

**Human analogy:** Your phone gets stolen while travelling.
1. Your environment gives you information → "my phone is stolen." *(interpret information)*
2. You conclude → "the thief may misuse my number." *(draw conclusion)*
3. You decide → "I'll call the telecom provider and block my number." *(make decision)*

That whole process is **reasoning** — and an AI agent has the same capability.

**Reasoning is needed in BOTH stages:**

**During Planning:**
- **Goal decomposition** — breaking the task into a series of steps requires reasoning.
- **Tool selection** — concluding *"this step needs Google Search, so I'll use that tool"* requires reasoning. *(E.g., the step "find salary for 2–4 years experience" needs an external tool.)*
- **Resource estimation** — estimating time, dependencies, and risks requires reasoning.

**During Execution:**
- **Decision-making** — when multiple options exist mid-execution, reasoning chooses one. *(E.g., interview 2 candidates or 3?)*
- **Human-in-the-loop judgment** — deciding *when* to do a task itself vs. *when* to ask a human. *(E.g., unsure about salary — search Google, or ask the human?)*
- **Error handling** — if a step fails (e.g., LinkedIn server is down), reasoning chooses: retry later, notify the human, or use another platform.

> Planning works *because* the agent has reasoning capability. Reasoning is needed everywhere across both stages — that's why it's a vital trait.

---

### 3.5 Adaptability

**Definition:** Adaptability is the agent's ability to **modify its plans, strategies, and actions in response to unexpected conditions**, all while staying aligned with the goal.

*Example:* When the AI Recruiter saw few applications after 3 days, it quickly adapted — proposing to run ads on LinkedIn and to modify the JD so Full-Stack Engineers could also apply.

**Reasons an agent must show adaptability:**

1. **Failures** — Agents work with many tools; a tool may fail. *E.g., the Calendar API is down, so the agent can't see your availability — it adapts by directly messaging you to ask.*
2. **External feedback from the environment** — Every agent operates in an **environment**:
   - A chess-playing agent → environment is the **chess board**.
   - A self-driving agent → environment is the **car, road, and pedestrians**.
   - The AI Recruiter → environment is **all applicants, LinkedIn, and you (the human)**.
   Sometimes the environment gives feedback that disrupts the flow (e.g., "very few people applied") → the agent must adapt its approach.
3. **Mid-way goal change** — If the goal itself changes (e.g., "hire a freelancer instead"), the agent must adapt because planning must change.

---

### 3.6 Context Awareness

**Definition:** Context awareness is the agent's ability to **understand, retain, and utilize relevant information** from the ongoing task, past interactions, user preferences, and environmental cues — to make better decisions throughout a multi-step process.

*Why it matters:* The hiring process can run for **many days**. If you ask the agent 4 days later "how many applicants applied?" and it has no retained context, it cannot function. Context awareness is essential.

**Types of context an Agentic AI application stores:**

| Context Type | Example |
|--------------|---------|
| **Original goal** | "Hire a Back-End Engineer" — must always be available. |
| **Progress + conversation history** | JD finalized and posted on LinkedIn; chat history between agent and human. |
| **Environment state** | Job posted on LinkedIn; 8 people have applied; ad budget runs out in 2 days. |
| **Tool responses** | Résumé parser: "Candidate B has 3 yrs Django + AWS." Calendar API: "Human free at 2 PM." |
| **User preferences** | Company prefers remote candidates; this human likes interview questions sent in a Google Doc. |
| **Policies / guardrails** | "Don't send the offer letter until approved"; "Never use platforms requiring paid ads on your own." |

**Memory** — Context awareness is implemented using **memory**. Agents generally have two types:

- **Short-Term Memory** — information about the **current session**: user messages, tool calls, immediate decisions.
  *Human analogy: "I'm shooting a video right now; topic is X; must finish by 4 PM."*
  *Agent example: résumé parser's response about Candidate B.*
- **Long-Term Memory** — high-level goals, past interactions, user preferences, and decisions **across sessions**.
  *Human analogy: "I live in Gurgaon; my job is X."*
  *Agent example: the guardrail "never send an offer letter without asking."*

---

## 4. The 5 Core Components of an Agentic AI System

These **five high-level components** appear in almost every Agentic AI application. (There can be more, but these are the most high-level.)

| # | Component | Body Analogy | Role |
|---|-----------|--------------|------|
| 1 | **Brain (LLM)** | The brain | Thinks, plans, reasons |
| 2 | **Orchestrator** | Nervous system / Project manager | Executes the plan |
| 3 | **Tools** | Hands and legs | Interacts with the external world |
| 4 | **Memory** | Memory | Stores context |
| 5 | **Supervisor** | — | Implements human-in-the-loop |

### 4.1 Brain
For an **LLM-based** Agentic AI system, the brain is generally the **LLM** itself. (Other agents exist, e.g., in Reinforcement Learning, but these notes focus on LLM-based agents.) The brain does the heavy lifting:
- **Goal interpretation** — figuring out what the user wants.
- **Planning** — breaking the goal into sub-goals.
- **Reasoning** — during both planning and execution.
- **Tool selection** — deciding which tool to use.
- **Natural language communication** — generating and understanding language between agent and human.

### 4.2 Orchestrator
The component that **executes the plan** step by step. The brain *plans*; the orchestrator *runs* the plan. Built using frameworks like **LangGraph, CrewAI, or AutoGen**. Its jobs:
- **Task sequencing** — the order in which steps execute.
- **Conditional routing** — based on a step's output, route to step 3 or step 4 (if/else logic).
- **Retry logic** — retry a step if a tool fails (e.g., LinkedIn was down).
- **Looping & iteration** — repeat steps when needed.
- **Delegation** — decide when to give a task to a human vs. the LLM.

> The orchestrator is like the **nervous system** — connected to every part — and acts like the **project manager** of the application.

### 4.3 Tools
Tools are how the agent **interacts with the external world** — API calls, database changes, sending emails, etc. **Tools are like hands and legs.**
A **knowledge base** (e.g., company documents provided via **RAG** — Retrieval-Augmented Generation) to ground responses with factual, domain-specific information is also considered a type of tool.

### 4.4 Memory
Memory performs several tasks:
- **Short-term memory** — current session: user messages, tool calls, immediate decisions.
- **Long-term memory** — high-level goals, past interactions, user preferences, cross-session decisions.
- **State tracking** — how much work is done and how much remains.

### 4.5 Supervisor
The component that implements the **human-in-the-loop** concept — it makes the agent and human work together. Useful when:
- **Approval needed** for high-risk actions (sending an offer letter, running paid ads) — it notifies the human and waits for permission.
- **Enforcing guardrails** on the system.
- **Handling edge cases / escalations** — *e.g., a guardrail says hire only candidates from IITs/NITs, but the agent finds a non-IIT/NIT candidate with an excellent résumé → it alerts the human to review.*

> This is a **high-level overview**. On a deeper dive, components contain sub-components — e.g., the Brain includes a **Planner** (generates multiple plans) and an **Evaluator** (evaluates those plans).

---

## 5. Quick Revision Summary

**What is Agentic AI?** A software paradigm where you give a system a *goal*, and it *plans* and *executes* to achieve it autonomously, with minimal human guidance. It is **proactive**, unlike **reactive** Generative AI chatbots.

**The 6 Key Characteristics** *(use these 6 questions to test if a system is "agentic" — all answers must be yes)*:

1. **Autonomy** — makes decisions & acts on its own; controlled via permission scope, human-in-the-loop, override controls, and guardrails.
2. **Goal-Oriented** — persistently works toward one objective; goals can be independent or constrained, are stored in memory, and can be altered mid-way.
3. **Planning** — breaks the goal into sub-goals via a 3-step process: *generate* multiple candidate plans → *evaluate* → *select*. Planning + Execution is an iterative loop.
4. **Reasoning** — interprets info, draws conclusions, makes decisions; needed in *both* planning and execution.
5. **Adaptability** — modifies plans in response to failures, environmental feedback, or goal changes.
6. **Context Awareness** — retains relevant info across a multi-step process; implemented via **short-term** and **long-term memory**.

**The 5 Core Components:**

| Component | One-line role |
|-----------|---------------|
| **Brain (LLM)** | Interprets goals, plans, reasons, selects tools, communicates |
| **Orchestrator** | Executes the plan — sequencing, routing, retries, looping, delegation |
| **Tools** | Interface to the external world (APIs, DB, email, RAG knowledge base) |
| **Memory** | Stores short-term, long-term context and tracks state |
| **Supervisor** | Implements human-in-the-loop — approvals, guardrails, escalations |

---

*Revision notes — Agentic AI Introduction (Playlist Video 2).*
