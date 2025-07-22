# AI Usage in Enhanced Dark Souls Knowledge Graph System

## Introduction

This document details how artificial intelligence (AI), particularly **Natural Language Processing (NLP)** and **Large Language Models (LLMs)**, is used in the Enhanced Dark Souls Knowledge Graph System.  
It covers architectural integration, model choices, NLP techniques, use cases, and practical examples.

---

## Table of Contents

1. [Overview of AI Components](#overview-of-ai-components)
2. [NLP Pipeline](#nlp-pipeline)
    - [Entity Recognition](#entity-recognition)
    - [Custom Pattern Matching](#custom-pattern-matching)
    - [Relationship Extraction](#relationship-extraction)
    - [Sentiment & Temporal Analysis](#sentiment--temporal-analysis)
    - [Advanced Features](#advanced-features)
3. [Knowledge Graph Construction](#knowledge-graph-construction)
4. [LLM Integration & Q&A](#llm-integration--qa)
    - [LangChain Agent](#langchain-agent)
    - [LLM Query Handling](#llm-query-handling)
5. [AI-Driven Export & Analysis](#ai-driven-export--analysis)
6. [AI Model Choices](#ai-model-choices)
7. [Performance Considerations](#performance-considerations)
8. [Example AI Workflows](#example-ai-workflows)
9. [Extending AI Capabilities](#extending-ai-capabilities)
10. [References](#references)

---

## Overview of AI Components

The system uses AI in two main areas:

1. **NLP for Data Extraction**
    - Spacy for entity, relation, and pattern extraction
    - NLTK for stopwords, tokenization, and lexical analysis

2. **LLM for Knowledge Exploration**
    - OpenAI's GPT models via LangChain
    - Conversational Q&A, search, and reasoning

---

## NLP Pipeline

### Entity Recognition

- Uses Spacy's pre-trained English model (`en_core_web_sm`) for **Named Entity Recognition (NER)**: PERSON, ORG, LOCATION, GPE, WORK_OF_ART, etc.
- Custom entities (weapons, locations) via pattern matchers.

### Custom Pattern Matching

- **Custom Spacy Matcher** for game-specific entities:
    - Weapon names: "sword", "axe", "dagger", etc.
    - Locations: "cathedral", "forest", "kingdom", etc.

### Relationship Extraction

- **Dependency parsing** to extract subject-verb-object triples.
- **Lore-specific verbs** (e.g., "banished", "wields", "sealed") for relationships unique to Dark Souls narrative.
- Item and faction relationships by keyword matching.

### Sentiment & Temporal Analysis

- **Lexicon-based sentiment**: Counts positive/negative words in context.
- **Temporal expressions**: Recognizes time indicators ("ancient", "before", "after") for story chronology.

### Advanced Features

- **Character attributes**: Adjective-Noun patterns for traits.
- **Hierarchical relationships**: Detects boss/minion, lord/knight, etc.
- **Co-occurrence patterns**: Finds entities appearing together.
- **Dialogue extraction**: Regex for quoted text.

---

## Knowledge Graph Construction

- **Entities** and **relationships** from NLP are nodes and edges in a NetworkX graph.
- Graph analytics (centrality, communities) use AI-extracted features.
- Graph cache and sampling for efficient analysis.

---

## LLM Integration & Q&A

### LangChain Agent

- Wraps the knowledge graph and tools for **natural language interaction**.
- Uses OpenAI's GPT-3.5/4 via LangChain for:
    - Interpreting user questions
    - Reasoning over graph structure
    - Generating explanations

### LLM Query Handling

- Accepts queries like "Who is the most important boss?" or "Show connections between Gwyn and Solaire".
- LLM parses intent, invokes knowledge graph analysis, and returns human-readable answers.

---

## AI-Driven Export & Analysis

- AI-generated analysis reports (statistics, summaries) in JSON.
- LLM provides context-aware commentary in interactive CLI.

---

## AI Model Choices

- **Spacy**: `en_core_web_sm` for fast, general-purpose English NLP.
- **NLTK**: For lexical tasks.
- **OpenAI GPT**: Chosen for conversational power and context understanding.
- **LangChain**: For agent memory and tool integration.

---

## Performance Considerations

- **Sampling** for centrality to avoid slow full-graph calculations.
- **Threading** for NLP and scraping.
- **Caching** of graph and analysis results.

---

## Example AI Workflows

### 1. Entity Discovery

> **Input:** "Find all knights in the game."
>
> **NLP:** Custom matcher identifies entities tagged as knights.
>
> **Output:** List of knight entities with categories.

### 2. Lore Relationship Reasoning

> **Input:** "What is the relationship between Gwyndolin and Anor Londo?"
>
> **LLM:** Parses the question, queries relationships in the graph, summarizes connections.

### 3. Story Chronology

> **Input:** "Describe the sequence of events leading to the linking of the fire."
>
> **NLP:** Temporal analysis extracts event order.
>
> **LLM:** Presents a narrative based on temporal patterns.

### 4. Community Detection

> **Input:** "Show the main factions and their members."
>
> **Graph:** Community algorithms cluster entities.
>
> **LLM:** Summarizes results and explains groupings.

---

## Extending AI Capabilities

- Add new Spacy matchers for more entity types (e.g., spells, artifacts)
- Integrate other LLMs (e.g., Claude, Gemini, local models)
- Add more advanced reasoning (multi-hop, analogical queries)
- Build a web or chat interface for real-time AI answers

---

## References

- [Spacy Documentation](https://spacy.io/usage)
- [NLTK Documentation](https://www.nltk.org/)
- [OpenAI GPT](https://platform.openai.com/)
- [LangChain](https://docs.langchain.com/)
- [NetworkX](https://networkx.org/documentation/stable/)

---

**For more details, see the code comments and the main README.**