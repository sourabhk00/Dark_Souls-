# Enhanced Dark Souls Knowledge Graph System

## Overview

This project is a **highly optimized, feature-rich pipeline** for extracting, processing, and analyzing knowledge from the [Dark Souls Wikidot](http://darksouls.wikidot.com/) wiki.  
It uses **advanced NLP**, **parallel data scraping**, **graph theory**, and **LLM-powered Q&A** to provide deep insights into Dark Souls lore, entities, relationships, and more.

It is perfect for:
- Lore explorers
- Data scientists
- Game designers
- AI/NLP researchers
- Fans of knowledge graphs and Dark Souls

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Architecture Diagram](#architecture-diagram)
3. [Setup Instructions](#setup-instructions)
4. [Directory Structure](#directory-structure)
5. [Module Breakdown](#module-breakdown)
6. [Core Features](#core-features)
7. [Command Reference](#command-reference)
8. [Exported Data Formats](#exported-data-formats)
9. [Performance Optimizations](#performance-optimizations)
10. [Advanced NLP Features](#advanced-nlp-features)
11. [Troubleshooting & FAQ](#troubleshooting--faq)
12. [Contributing](#contributing)
13. [License & Credits](#license--credits)

---

## Project Goals

- **Scrape** all items, characters, locations, and lore from Dark Souls Wikidot, by category and subcategory.
- **Extract** structured entities and relationships using advanced NLP and custom patterns.
- **Build** a robust knowledge graph using NetworkX, with centrality and community analysis.
- **Enable** natural-language Q&A via LangChain and OpenAI LLM.
- **Export** all processed data in multiple formats (Excel, CSV, JSON, ZIP).
- **Visualize** entity relationships and graph statistics interactively.

---

## Architecture Diagram

```
[Web Scraping] --> [Raw Data Storage] --> [Advanced NLP Processing]
       |                   |                   |
       V                   V                   V
[Category & Subcategory]   [entities.csv, relationships.csv, advanced_features.json]
       |                   |                   |
       |                   V                   |
       |-----------[Knowledge Graph]-----------|
                           |
                  [LLM Q&A System]      [Visualization]
                           |               |
                       [Export Tools]  [CLI Interface]
```

---

## Setup Instructions

### 1. Clone & Install

```sh
git clone https://github.com/yourusername/enhanced_darksouls_kg.git
cd enhanced_darksouls_kg
pip install -r requirements.txt
```

### 2. NLTK Data

The script will automatically download NLTK data (stopwords, punkt, etc.).  
If you see missing data errors, run:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### 3. Run Main Script

```sh
python main.py
```

- If data exists, it loads it for fast startup.
- If not, it scrapes, processes, and builds everything automatically.

---

## Directory Structure

```
enhanced_darksouls_kg/
│
├── darksouls_data/           # Raw scraped CSVs by category/subcategory
├── knowledge_graph/          # Processed entities, relationships, graph cache, advanced_features.json
├── exports/                  # Multi-format exports (Excel, CSV, JSON, ZIP)
│
├── main.py                   # Main script & CLI entrypoint
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Module Breakdown

### 1. Imports & Dependencies

Handles all major needs:
- **Web scraping:** requests, BeautifulSoup, urllib
- **Data wrangling:** pandas, numpy, os, re, json, pickle
- **NLP:** spacy (with custom matchers), nltk
- **Concurrency:** ThreadPoolExecutor, threading
- **Graph analysis:** networkx
- **Visualization:** matplotlib
- **LLM agent:** langchain, openai

### 2. NLP Initialization & Patterns

- Loads Spacy English model (`en_core_web_sm`)
- Downloads and sets up NLTK stopwords, tokenizers
- Adds custom Spacy matcher patterns for weapons and locations

### 3. Category & Subcategory Definitions

All main Dark Souls game categories are mapped to their respective Wikidot URLs:
- Weapons, Armor, Spells, NPCs, Bosses, Areas, etc.

### 4. Data Extraction & Scraping

- `extract_links_by_subcategory(url)`: Parses subcategory links from category pages
- `scrape_all_data()`: Scrapes all categories/subcategories in parallel
- `scrape_page_content(url)`: Scrapes, cleans, and returns page text

### 5. Advanced NLP Processing

Functions for:
- **Entity extraction:** Standard NER + custom weapons/locations
- **Relationship extraction:** Subject-verb-object, lore verbs, items, factions, hierarchy
- **Sentiment & temporal analysis**
- **Co-occurrence and dialogue**
- **Character attributes, environment mentions**

### 6. Export Functionality

`create_comprehensive_export()`:  
Creates a timestamped export directory with:
- Excel file (multiple sheets)
- Category-wise and relationship-wise CSVs
- Analysis JSON reports
- Raw scraped data
- All data zipped

### 7. Knowledge Graph Analysis

`KnowledgeGraphAnalyzer` class:
- Loads/builds/caches a NetworkX graph
- Centrality analysis (betweenness, closeness, degree, pagerank, eigenvector)
- Graph statistics (density, clustering, components)
- Community detection
- Shortest path and entity relationship analysis

### 8. LLM & Agent Integration

- `KnowledgeGraphTool`: LangChain-compatible tool for querying the graph
- `DarkSoulsQASystem`: Integrates everything with LLM, memory, and agent for natural language Q&A

### 9. CLI Interactive Q&A System

`start_interactive_session()`:  
User can ask questions, search, analyze, export, and interact via simple commands.

### 10. Visualization

`visualize_graph_sample(analyzer, max_nodes)`:  
Creates and saves a colored network graph visualization for most-connected entities.

### 11. Main Pipeline

- Loads or builds all data
- Runs advanced NLP
- Initializes graph and LLM agent
- Provides statistics, visualizations, and starts interactive CLI

---

## Core Features

- **Parallel scraping** for speed
- **Custom NLP** for game-specific entities
- **Graph analytics**: Centrality, communities, statistics, shortest paths
- **Natural language Q&A** via LLM
- **Comprehensive exports** to Excel, CSV, JSON, ZIP
- **Graph visualization** with entity-type coloring
- **Modular, extensible codebase**

---

## Command Reference

| Command           | Description                                             |
|-------------------|--------------------------------------------------------|
| `help`            | Show all available commands                            |
| `search [name]`   | Search for an entity (fuzzy match)                     |
| `characters`      | List all character entities                            |
| `characters [x]`  | Search for specific characters                         |
| `stats`           | Show graph statistics                                  |
| `communities`     | Detect and list graph communities                      |
| `export`          | Create/export all data                                 |
| `quit`            | Exit the system                                        |

---

## Exported Data Formats

- **Excel**: Entities, relationships, summary sheets
- **CSV**: Entities by category, relationships by type
- **JSON**: Analysis reports, advanced NLP features
- **ZIP**: All above data for easy download

---

## Performance Optimizations

- **ThreadPoolExecutor** for concurrent scraping and processing
- **Deduplication** of entities and relationships
- **Efficient pandas types and chunking**
- **Graph caching** (pickle)
- **Sampling for expensive graph metrics**
- **Avoid redundant visualizations**

---

## Advanced NLP Features

- **Custom entity recognition**: Weapons, locations, items, factions, environments
- **Sentiment analysis** for lore context
- **Temporal expression extraction** for story chronology
- **Hierarchical relationships** (boss/minion, lord/knight)
- **Attribute and co-occurrence extraction** for richer graphs
- **Dialogue detection**

---

## Troubleshooting & FAQ

- **NLTK errors:** Run provided NLTK download commands
- **Spacy model missing:** Run `python -m spacy download en_core_web_sm`
- **Scraping errors:** Check your internet, Wikidot may block requests; try increasing random wait time in scraping
- **Out-of-memory:** Reduce max_workers or chunk sizes in scraping/graph building

---

## Contributing

1. Fork the repo, clone your fork
2. Create a feature branch
3. Add your code, tests, docs
4. Open a PR describing your changes!

**Ideas:** New entity patterns, other game wikis, more graph algorithms, web dashboard, etc.

---

## License & Credits

- **Open source** for research and educational use
- **Game content**: Property of FromSoftware/Bandai Namco
- **Libraries**: Spacy, NLTK, NetworkX, LangChain, OpenAI

---

## Acknowledgments

Special thanks to:
- [Dark Souls Wikidot](http://darksouls.wikidot.com/)
- [Spacy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [NetworkX](https://networkx.org/)
- [LangChain](https://www.langchain.com/)

---

**May the flames guide thee!**