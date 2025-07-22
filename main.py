# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import os
import re
import spacy
from spacy.matcher import Matcher
import zipfile
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords
import nltk
import time
import random
import json
from typing import List, Dict, Any, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# main.py
from langgraph import LangGraph

def main():
    # Your logic using LangGraph
    graph = LangGraph()
    # Additional code...

if __name__ == "__main__":
    main()from langgraph import LangGraph  # Ensure LangGraph is installed and imported

# Initialize NLP
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Function to create a LangGraph
def create_langgraph(entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> LangGraph:
    """Creates a LangGraph instance from entities and relationships."""
    graph = LangGraph()
    
    # Add nodes and edges from the DataFrames
    for _, row in entities_df.iterrows():
        graph.add_node(row['entity'], type=row['type'], category=row['category'])

    for _, row in relationships_df.iterrows():
        graph.add_edge(row['subject'], row['object'], predicate=row['predicate'])

    return graph

def visualize_graph(graph: LangGraph):
    """Visualizes the LangGraph."""
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph.get_nx_graph())
    nx.draw(graph.get_nx_graph(), pos, node_size=700, with_labels=True)
    plt.title("LangGraph Visualization")
    plt.show()

def main():
    """Main function to run the Dark Souls Knowledge Graph system."""
    # Load existing data
    entities_df, relationships_df = load_existing_data()
    
    # Create the LangGraph
    lang_graph = create_langgraph(entities_df, relationships_df)

    # Visualize the created graph
    visualize_graph(lang_graph)

    # Additional tasks...
    print("🔍 Gathering extra data...")
    all_links_by_category = scrape_all_data()
    
    print("✅ All tasks completed!")

if __name__ == "__main__":
    main()
# Step 2: Import necessary libraries
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import os
import re
import spacy
from spacy.matcher import Matcher
import zipfile
from datetime import datetime

import numpy as np

from nltk.corpus import stopwords
import nltk
import time
import random
import json
from typing import List, Dict, Any, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain and LLM imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool

# Initialize NLP
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Add custom patterns for better entity recognition
weapon_patterns = [
    [{"LOWER": {"IN": ["sword", "blade", "spear", "axe", "bow", "crossbow", "halberd", "dagger", "club", "hammer"]}},
     {"POS": "NOUN", "OP": "?"}],
    [{"POS": "ADJ", "OP": "?"}, {"LOWER": {"IN": ["sword", "blade", "weapon", "armor", "shield"]}}, {"POS": "NOUN", "OP": "?"}]
]

location_patterns = [
    [{"LOWER": {"IN": ["cathedral", "forest", "catacombs", "depths", "archives", "asylum", "fortress"]}},
     {"POS": "NOUN", "OP": "?"}],
    [{"POS": "ADJ", "OP": "?"}, {"LOWER": {"IN": ["kingdom", "realm", "city", "village", "ruins", "temple"]}}, {"POS": "NOUN", "OP": "?"}]
]

matcher.add("WEAPON", weapon_patterns)
matcher.add("LOCATION", location_patterns)

# Step 3: Define all categories with their URLs
category_urls = {
    "Weapons": "http://darksouls.wikidot.com/weapons",
    "Shields": "http://darksouls.wikidot.com/shields",
    "Spell Tools": "http://darksouls.wikidot.com/spell-tools",
    "Weapon Upgrades": "http://darksouls.wikidot.com/upgrade",
    "Armor Sets": "http://darksouls.wikidot.com/armor",
    "Head Armor": "http://darksouls.wikidot.com/head",
    "Chest Armor": "http://darksouls.wikidot.com/chest",
    "Hands Armor": "http://darksouls.wikidot.com/hands",
    "Legs Armor": "http://darksouls.wikidot.com/legs",
    "Ammo": "http://darksouls.wikidot.com/ammo",
    "Bonfire Items": "http://darksouls.wikidot.com/bonfire-items",
    "Consumables": "http://darksouls.wikidot.com/consumables",
    "Multiplayer Items": "http://darksouls.wikidot.com/multiplayer-items",
    "Rings": "http://darksouls.wikidot.com/rings",
    "Keys": "http://darksouls.wikidot.com/keys",
    "Pyromancies": "http://darksouls.wikidot.com/pyromancies",
    "Sorceries": "http://darksouls.wikidot.com/sorceries",
    "Miracles": "http://darksouls.wikidot.com/miracles",
    "Story": "http://darksouls.wikidot.com/story",
    "Bosses": "http://darksouls.wikidot.com/bosses",
    "Enemies": "http://darksouls.wikidot.com/enemies",
    "Merchants": "http://darksouls.wikidot.com/merchants",
    "NPCs": "http://darksouls.wikidot.com/npcs",
    "Areas": "http://darksouls.wikidot.com/areas",
    "Stats": "http://darksouls.wikidot.com/stats",
    "Classes": "http://darksouls.wikidot.com/classes",
    "Gifts": "http://darksouls.wikidot.com/gifts",
    "Covenants": "http://darksouls.wikidot.com/covenants",
    "Trophies": "http://darksouls.wikidot.com/trophies",
    "Achievements": "http://darksouls.wikidot.com/achievements"
}

# Step 4: Function to extract subcategory-wise links
def extract_links_by_subcategory(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        content = soup.find('div', id='page-content')

        subcategory_links = {}
        current_heading = "General"

        for tag in content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'a']):
            if tag.name in ['h1', 'h2', 'h3'] and tag.get_text(strip=True):
                current_heading = tag.get_text(strip=True)
                if current_heading not in subcategory_links:
                    subcategory_links[current_heading] = []

            if tag.name == 'a' and tag.get('href'):
                href = tag['href'].strip()
                if href.startswith(('#', 'javascript:', 'mailto:', 'void(0)')):
                    continue
                full_url = urljoin(url, href)
                if "darksouls.wikidot.com" in full_url:
                    subcategory_links.setdefault(current_heading, []).append(full_url)

        for k in subcategory_links:
            subcategory_links[k] = sorted(set(subcategory_links[k]))

        return subcategory_links
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return {}

# Step 5: Extract all links with subcategories (with threading for speed)
def scrape_all_data():
    all_links_by_category = {}
    print("🔍 Extracting category links with parallel processing...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_category = {
            executor.submit(extract_links_by_subcategory, url): category 
            for category, url in category_urls.items()
        }
        
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                result = future.result()
                all_links_by_category[category] = result
                print(f"  ✅ Completed: {category}")
            except Exception as e:
                print(f"  ❌ Failed: {category} - {e}")
                all_links_by_category[category] = {}
    
    return all_links_by_category

# Step 6: Scrape page content function (optimized)
def scrape_page_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        time.sleep(random.uniform(0.3, 0.8))  # Reduced delay for faster scraping
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        content_div = soup.find('div', id='page-content')

        if content_div:
            # Clean up content
            for element in content_div.find_all(['div', 'script', 'style', 'table', 'img']):
                element.decompose()

            text = content_div.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive newlines
            return text
        return ""
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

# Step 7: Advanced NLP processing functions
lore_verbs = [
    "created", "wields", "banished", "fought", "follower", "worships", "cursed", 
    "defeated", "descendant", "betrayed", "guarded", "protects", "summoned", "linked",
    "resides", "haunts", "guards", "owns", "uses", "slain", "aided", "opposes",
    "forged", "discovered", "destroyed", "corrupted", "sealed", "awakened", "imprisoned"
]

item_keywords = ["sword", "ring", "armor", "pyromancy", "miracle", "talisman", "weapon", "shield", "key", "soul",
                "blade", "spear", "bow", "crossbow", "halberd", "dagger", "club", "hammer", "catalyst", "pendant"]

faction_keywords = ["covenant", "faction", "order", "followers", "knights", "disciples", "legion", "clan", "guild",
                   "brotherhood", "sisterhood", "company", "guard", "army", "cult"]

environment_keywords = ["ruins", "catacombs", "abyss", "kingdom", "fortress", "depths", "archives", "blight", "lake",
                       "forest", "cathedral", "asylum", "undead", "shrine", "temple", "tower", "bridge", "garden"]

# Advanced NLP Functions
def extract_sentiment(text):
    """Extract sentiment from text using basic lexicon approach"""
    positive_words = ['triumph', 'victory', 'glory', 'honor', 'noble', 'brave', 'sacred', 'divine', 'blessed']
    negative_words = ['cursed', 'dark', 'evil', 'corrupt', 'fallen', 'doomed', 'despair', 'death', 'hollow']
    
    doc = nlp(text.lower())
    pos_count = sum(1 for token in doc if token.text in positive_words)
    neg_count = sum(1 for token in doc if token.text in negative_words)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

def extract_temporal_expressions(text):
    """Extract temporal expressions and sequences"""
    doc = nlp(text)
    temporal_patterns = []
    
    # Look for temporal indicators
    temporal_words = ['before', 'after', 'during', 'ancient', 'old', 'first', 'last', 'original', 'final']
    
    for sent in doc.sents:
        for token in sent:
            if token.text.lower() in temporal_words:
                temporal_patterns.append({
                    'temporal_word': token.text,
                    'context': sent.text.strip(),
                    'entities_in_context': [ent.text for ent in sent.ents]
                })
    
    return temporal_patterns

def extract_character_attributes(text):
    """Extract character attributes and descriptions"""
    doc = nlp(text)
    attributes = []
    
    # Look for descriptive patterns
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            # Find if the noun is likely a character
            if any(ent.label_ == "PERSON" and token.head.text in ent.text for ent in doc.ents):
                attributes.append({
                    'character': token.head.text,
                    'attribute': token.text,
                    'context': token.sent.text.strip()
                })
    
    return attributes

def extract_co_occurrence_patterns(text):
    """Extract entity co-occurrence patterns"""
    doc = nlp(text)
    co_occurrences = []
    
    for sent in doc.sents:
        entities = [ent.text for ent in sent.ents]
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    co_occurrences.append({
                        'entity1': entities[i],
                        'entity2': entities[j],
                        'context': sent.text.strip(),
                        'distance': abs(j - i)
                    })
    
    return co_occurrences

def extract_hierarchical_relationships(text):
    """Extract hierarchical relationships (boss-subordinate, lord-knight, etc.)"""
    doc = nlp(text)
    hierarchical_rels = []
    
    hierarchy_patterns = [
        ('lord', 'knight'), ('king', 'subject'), ('master', 'servant'),
        ('boss', 'minion'), ('leader', 'follower'), ('god', 'worshipper')
    ]
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        for superior, subordinate in hierarchy_patterns:
            if superior in sent_text and subordinate in sent_text:
                entities = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
                if len(entities) >= 2:
                    hierarchical_rels.append({
                        'type': 'hierarchical',
                        'superior_role': superior,
                        'subordinate_role': subordinate,
                        'entities': entities,
                        'context': sent.text.strip()
                    })
    
    return hierarchical_rels

def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_informative(word):
    return len(word) >= 3 and word.lower() not in stop_words and not word.isnumeric()

def advanced_entity_filter(ents):
    filtered = []
    seen = set()
    for ent in ents:
        name = ent.text.strip()
        if not is_informative(name) or name.lower() in seen:
            continue
        seen.add(name.lower())
        filtered.append((name, ent.label_))
    return filtered

def extract_entities(text):
    doc = nlp(text)
    
    # Standard NER entities
    ents = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'WORK_OF_ART']]
    
    # Custom pattern matching for weapons and locations
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        ents.append(type('Entity', (), {'text': span.text, 'label_': label})())
    
    return advanced_entity_filter(ents)

def extract_relationships(text):
    doc = nlp(text)
    rels = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                obj = [w for w in token.rights if w.dep_ in ("dobj", "attr", "prep", "pobj")]
                if subj and obj:
                    rels.append((subj[0].text, token.lemma_, obj[0].text))
    return rels

def extract_lore_relationships(text):
    doc = nlp(text)
    lore_relations = []
    for sent in doc.sents:
        for token in sent:
            if token.lemma_ in lore_verbs and token.pos_ == "VERB":
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                obj = [w for w in token.rights if w.dep_ in ("dobj", "attr", "prep", "pobj")]
                if subj and obj:
                    lore_relations.append((subj[0].text, token.lemma_, obj[0].text))
    return lore_relations

def extract_items(text):
    doc = nlp(text)
    mentions = []
    for token in doc:
        if token.lemma_.lower() in item_keywords:
            mentions.append(token.text)
    return list(set(mentions))

def extract_item_lore(text):
    doc = nlp(text)
    item_rels = []
    for sent in doc.sents:
        for token in sent:
            if token.lemma_.lower() in item_keywords:
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "poss")]
                if subj:
                    item_rels.append((subj[0].text, "has_item", token.text))
    return item_rels

def extract_environment_mentions(text):
    doc = nlp(text)
    return list(set([token.text for token in doc if token.lemma_.lower() in environment_keywords]))

def extract_faction_relationships(text):
    doc = nlp(text)
    faction_rels = []
    for sent in doc.sents:
        for token in sent:
            if token.lemma_.lower() in faction_keywords:
                subj = [w for w in token.lefts if w.dep_ in ("nsubj", "poss")]
                if subj:
                    faction_rels.append((subj[0].text, "member_of", token.text))
    return faction_rels

def extract_inferable_relationships(text):
    inferable = []
    sentences = list(nlp(text).sents)
    for i in range(1, len(sentences)):
        prev_sent = sentences[i-1].text
        curr_sent = sentences[i].text
        if any(p in curr_sent for p in ["he", "she", "they", "him", "her", "his", "their"]):
            inferable.append((clean_text(prev_sent), "->", clean_text(curr_sent)))
    return inferable

def extract_dialogue(text):
    dialogues = re.findall(r'["""''\'](.*?)["""''\']', text)
    return [dlg.strip() for dlg in dialogues if len(dlg.strip()) > 10]

# Export functionality
def create_comprehensive_export():
    """Create comprehensive export package with all data formats"""
    print("📦 Creating comprehensive export package...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = f"exports/darksouls_complete_{timestamp}"
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # Load all data
        entities_df = pd.read_csv("knowledge_graph/entities.csv")
        relationships_df = pd.read_csv("knowledge_graph/relationships.csv")
        
        # 1. Create Excel files with multiple sheets
        print("📊 Creating Excel exports...")
        with pd.ExcelWriter(f"{export_dir}/darksouls_complete_data.xlsx", engine='openpyxl') as writer:
            entities_df.to_excel(writer, sheet_name='Entities', index=False)
            relationships_df.to_excel(writer, sheet_name='Relationships', index=False)
            
            # Category summaries
            category_summary = entities_df.groupby('category').agg({
                'entity': 'count',
                'type': lambda x: x.value_counts().to_dict()
            }).rename(columns={'entity': 'entity_count'})
            category_summary.to_excel(writer, sheet_name='Category_Summary')
            
            # Relationship type summary
            rel_summary = relationships_df.groupby('type').agg({
                'subject': 'count',
                'predicate': lambda x: x.value_counts().head(10).to_dict()
            }).rename(columns={'subject': 'relationship_count'})
            rel_summary.to_excel(writer, sheet_name='Relationship_Summary')
        
        # 2. Create category-wise CSV files
        print("📁 Creating category-wise CSV exports...")
        categories_dir = f"{export_dir}/entities_by_category"
        os.makedirs(categories_dir, exist_ok=True)
        
        for category in entities_df['category'].unique():
            category_entities = entities_df[entities_df['category'] == category]
            safe_name = re.sub(r'[^\w\-_]', '_', category)
            category_entities.to_csv(f"{categories_dir}/{safe_name}_entities.csv", index=False)
        
        # 3. Create relationship-wise CSV files
        print("🔗 Creating relationship-wise CSV exports...")
        relationships_dir = f"{export_dir}/relationships_by_type"
        os.makedirs(relationships_dir, exist_ok=True)
        
        for rel_type in relationships_df['type'].unique():
            type_rels = relationships_df[relationships_df['type'] == rel_type]
            type_rels.to_csv(f"{relationships_dir}/{rel_type}_relationships.csv", index=False)
        
        # 4. Copy raw scraped data
        print("📄 Copying raw scraped data...")
        if os.path.exists("darksouls_data"):
            raw_data_dir = f"{export_dir}/raw_scraped_data"
            os.makedirs(raw_data_dir, exist_ok=True)
            for file in os.listdir("darksouls_data"):
                if file.endswith('.csv'):
                    pd.read_csv(f"darksouls_data/{file}").to_csv(f"{raw_data_dir}/{file}", index=False)
        
        # 5. Create analysis reports
        print("📈 Creating analysis reports...")
        analysis_dir = f"{export_dir}/analysis_reports"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Entity statistics
        entity_stats = {
            'total_entities': len(entities_df),
            'entities_by_type': entities_df['type'].value_counts().to_dict(),
            'entities_by_category': entities_df['category'].value_counts().to_dict(),
            'most_common_entities': entities_df['entity'].value_counts().head(20).to_dict()
        }
        
        with open(f"{analysis_dir}/entity_statistics.json", 'w') as f:
            json.dump(entity_stats, f, indent=2)
        
        # Relationship statistics
        rel_stats = {
            'total_relationships': len(relationships_df),
            'relationships_by_type': relationships_df['type'].value_counts().to_dict(),
            'most_common_predicates': relationships_df['predicate'].value_counts().head(20).to_dict(),
            'relationships_by_category': relationships_df['category'].value_counts().to_dict()
        }
        
        with open(f"{analysis_dir}/relationship_statistics.json", 'w') as f:
            json.dump(rel_stats, f, indent=2)
        
        # 6. Create ZIP file
        print("🗜️ Creating ZIP archive...")
        zip_filename = f"darksouls_complete_export_{timestamp}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Export complete! Created:")
        print(f"   📁 Directory: {export_dir}")
        print(f"   🗜️ ZIP file: {zip_filename}")
        print(f"   📊 Excel file: darksouls_complete_data.xlsx")
        print(f"   📈 Analysis reports in: analysis_reports/")
        
        return export_dir, zip_filename
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None, None

# NEW: Enhanced Knowledge Graph Analysis Class
class KnowledgeGraphAnalyzer:
    def __init__(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame):
        self.entities_df = entities_df
        self.relationships_df = relationships_df
        self.graph = self._load_or_build_graph()
        self._centrality_cache = {}
        self._stats_cache = None
        self._community_cache = None
        
    def _load_or_build_graph(self) -> nx.Graph:
        """Load cached graph or build new one"""
        graph_cache_path = "knowledge_graph/graph_cache.pkl"
        
        try:
            # Check if cache exists and is newer than CSV files
            import os
            if os.path.exists(graph_cache_path):
                cache_time = os.path.getmtime(graph_cache_path)
                entities_time = os.path.getmtime("knowledge_graph/entities.csv")
                relationships_time = os.path.getmtime("knowledge_graph/relationships.csv")
                
                if cache_time > entities_time and cache_time > relationships_time:
                    print("📦 Loading cached graph...")
                    with open(graph_cache_path, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
        
        print("🔧 Building new graph...")
        G = self._build_networkx_graph()
        
        # Cache the graph
        try:
            with open(graph_cache_path, 'wb') as f:
                pickle.dump(G, f)
            print("💾 Graph cached successfully")
        except Exception as e:
            print(f"⚠️ Graph caching failed: {e}")
        
        return G
    
    def _build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph from entities and relationships with optimizations"""
        G = nx.Graph()
        
        # Add nodes (entities) - vectorized approach
        print("🔗 Adding nodes...")
        entity_data = [(row['entity'], {
            'type': row['type'],
            'category': row['category'], 
            'subcategory': row['subcategory']
        }) for _, row in self.entities_df.iterrows()]
        G.add_nodes_from(entity_data)
        
        # Add edges (relationships) - filtered for existing nodes only
        print("🔗 Adding edges...")
        valid_nodes = set(G.nodes())
        edge_data = []
        
        # Process in chunks for better memory usage
        chunk_size = 1000
        for i in range(0, len(self.relationships_df), chunk_size):
            chunk = self.relationships_df.iloc[i:i+chunk_size]
            for _, rel in chunk.iterrows():
                if rel['subject'] in valid_nodes and rel['object'] in valid_nodes:
                    edge_data.append((rel['subject'], rel['object'], {
                        'predicate': rel['predicate'],
                        'rel_type': rel['type'],
                        'category': rel['category']
                    }))
        
        G.add_edges_from(edge_data)
        print(f"🎯 Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    @lru_cache(maxsize=1)
    def get_entity_centrality(self) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures for entities with caching and sampling"""
        if self._centrality_cache:
            return self._centrality_cache
            
        print("🔄 Calculating centrality measures...")
        
        # Always use sampling for faster computation
        sample_size = min(1000, self.graph.number_of_nodes())
        sample_nodes = list(self.graph.nodes())[:sample_size]
        subgraph = self.graph.subgraph(sample_nodes)
        
        self._centrality_cache = {
            'betweenness': nx.betweenness_centrality(subgraph, k=min(100, sample_size)),
            'closeness': nx.closeness_centrality(subgraph),
            'degree': nx.degree_centrality(self.graph),  # Fast for full graph
            'pagerank': nx.pagerank(subgraph, max_iter=50)
        }
        
        # Add eigenvector centrality with error handling
        try:
            self._centrality_cache['eigenvector'] = nx.eigenvector_centrality(subgraph, max_iter=100)
        except:
            self._centrality_cache['eigenvector'] = {}
        
        return self._centrality_cache
    
    def detect_communities(self):
        """Detect communities in the graph using fast algorithms"""
        if self._community_cache is not None:
            return self._community_cache
        
        print("🏘️ Detecting communities...")
        try:
            # Use fast community detection
            communities = nx.community.greedy_modularity_communities(self.graph)
            self._community_cache = [list(community) for community in communities]
            print(f"Found {len(self._community_cache)} communities")
        except Exception as e:
            print(f"Community detection failed: {e}")
            self._community_cache = []
        
        return self._community_cache
    
    def analyze_graph_motifs(self):
        """Analyze common graph motifs and patterns"""
        print("🔍 Analyzing graph motifs...")
        
        # Triangle count (3-cliques)
        triangles = sum(nx.triangles(self.graph).values()) // 3
        
        # 4-cliques
        cliques_4 = sum(1 for clique in nx.find_cliques(self.graph) if len(clique) == 4)
        
        # Bridge edges
        bridges = list(nx.bridges(self.graph))
        
        # Articulation points
        articulation_points = list(nx.articulation_points(self.graph))
        
        return {
            'triangles': triangles,
            '4_cliques': cliques_4,
            'bridges': len(bridges),
            'articulation_points': len(articulation_points),
            'bridge_list': bridges[:10],  # Show first 10
            'critical_nodes': articulation_points[:10]  # Show first 10
        }
    
    def find_shortest_path(self, entity1: str, entity2: str) -> List[str]:
        """Find shortest path between two entities"""
        try:
            return nx.shortest_path(self.graph, entity1, entity2)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_connected_components(self) -> List[List[str]]:
        """Find connected components in the graph"""
        return [list(component) for component in nx.connected_components(self.graph)]
    
    def analyze_entity_relationships(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        """Analyze relationships for a specific entity with enhanced features"""
        if entity not in self.graph.nodes:
            return {"error": f"Entity '{entity}' not found in graph"}
        
        # Get neighbors at different depths
        neighbors = {}
        for d in range(1, depth + 1):
            neighbors[f"depth_{d}"] = list(nx.single_source_shortest_path_length(self.graph, entity, cutoff=d).keys())
        
        # Get direct relationships
        direct_rels = self.relationships_df[
            (self.relationships_df['subject'] == entity) | 
            (self.relationships_df['object'] == entity)
        ]
        
        # Calculate local clustering coefficient
        clustering = nx.clustering(self.graph, entity)
        
        # Get centrality measures for this entity
        centrality_data = self.get_entity_centrality()
        entity_centrality = {}
        for measure, values in centrality_data.items():
            entity_centrality[measure] = values.get(entity, 0)
        
        return {
            "entity": entity,
            "neighbors": neighbors,
            "direct_relationships": direct_rels.to_dict('records'),
            "degree": self.graph.degree[entity],
            "clustering_coefficient": clustering,
            "centrality_measures": entity_centrality,
            "relationship_types": direct_rels['type'].value_counts().to_dict()
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics with caching"""
        if self._stats_cache is not None:
            return self._stats_cache
        
        print("📊 Calculating comprehensive graph statistics...")
        
        # Basic stats
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # Connected components
        components = list(nx.connected_components(self.graph))
        num_components = len(components)
        largest_component_size = len(max(components, key=len)) if components else 0
        
        # Clustering
        avg_clustering = nx.average_clustering(self.graph)
        
        # Degree distribution
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Path lengths (only for largest component if graph is disconnected)
        if nx.is_connected(self.graph):
            avg_path_length = nx.average_shortest_path_length(self.graph)
            diameter = nx.diameter(self.graph)
        else:
            largest_cc = max(components, key=len) if components else []
            if len(largest_cc) > 1:
                largest_subgraph = self.graph.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(largest_subgraph)
                diameter = nx.diameter(largest_subgraph)
            else:
                avg_path_length = "Graph is disconnected"
                diameter = "Graph is disconnected"
        
        # Graph motifs
        motifs = self.analyze_graph_motifs()
        
        self._stats_cache = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "num_connected_components": num_components,
            "largest_component_size": largest_component_size,
            "average_clustering": avg_clustering,
            "average_degree": avg_degree,
            "max_degree": max_degree,
            "average_shortest_path_length": avg_path_length,
            "diameter": diameter,
            "motifs": motifs
        }
        
        return self._stats_cache
    
    def create_interactive_graph(self, category: str = None, max_nodes: int = 100):
        """Create an interactive graph visualization"""
        print(f"🎨 Creating interactive graph visualization...")
        
        if category:
            # Filter by category
            category_entities = self.entities_df[self.entities_df['category'] == category]
            nodes_to_include = set(category_entities['entity'].tolist())
            subgraph = self.graph.subgraph(nodes_to_include)
            title = f"Dark Souls - {category} Network"
        else:
            # Use most connected nodes for full graph
            centrality = self.get_entity_centrality()['degree']
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes_to_include = [node[0] for node in top_nodes]
            subgraph = self.graph.subgraph(nodes_to_include)
            title = f"Dark Souls Knowledge Graph (Top {max_nodes} Connected)"
        
        plt.figure(figsize=(20, 16))
        
        # Use spring layout with optimized parameters
        pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        
        # Color nodes by type
        entity_types = {}
        for node in subgraph.nodes():
            entity_row = self.entities_df[self.entities_df['entity'] == node]
            if not entity_row.empty:
                entity_types[node] = entity_row.iloc[0]['type']
            else:
                entity_types[node] = 'UNKNOWN'
        
        type_colors = {
            'PERSON': '#FF6B6B', 'ORG': '#4ECDC4', 'LOC': '#45B7D1', 
            'GPE': '#FFA07A', 'WORK_OF_ART': '#98D8C8', 'WEAPON': '#F7DC6F',
            'LOCATION': '#BB8FCE', 'UNKNOWN': '#BDC3C7'
        }
        
        # Draw nodes by type
        for entity_type, color in type_colors.items():
            nodes_of_type = [node for node, ntype in entity_types.items() if ntype == entity_type]
            if nodes_of_type:
                # Size nodes by degree
                node_sizes = [self.graph.degree[node] * 50 + 100 for node in nodes_of_type]
                nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes_of_type, 
                                     node_color=color, node_size=node_sizes, 
                                     alpha=0.8, label=f"{entity_type} ({len(nodes_of_type)})")
        
        # Draw edges with varying thickness
        edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray', 
                              width=[w*0.5 for w in edge_weights])
        
        # Draw labels for high-degree nodes only
        high_degree_nodes = [node for node in subgraph.nodes() if self.graph.degree[node] >= 3]
        high_degree_pos = {node: pos[node] for node in high_degree_nodes if node in pos}
        nx.draw_networkx_labels(subgraph, high_degree_pos, font_size=8, font_weight='bold')
        
        plt.title(title, size=20, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        
        # Save with high quality
        filename = f"knowledge_graph/{category.replace(' ', '_').lower() if category else 'interactive'}_network.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Interactive graph saved to {filename}")
        plt.show()
        
        return subgraph
    
    def create_category_graph(self, category: str) -> None:
        """Create a graph visualization for a specific category"""
        category_entities = self.entities_df[self.entities_df['category'] == category]
        category_nodes = category_entities['entity'].tolist()
        
        if len(category_nodes) == 0:
            print(f"⚠️ No entities found for category: {category}")
            return
        
        # Create subgraph
        subgraph = self.graph.subgraph(category_nodes)
        
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Color nodes by subcategory
        subcategories = category_entities['subcategory'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(subcategories)))
        color_map = dict(zip(subcategories, colors))
        
        for subcat in subcategories:
            subcat_nodes = category_entities[category_entities['subcategory'] == subcat]['entity'].tolist()
            nx.draw_networkx_nodes(subgraph, pos, nodelist=subcat_nodes, 
                                 node_color=[color_map[subcat]], node_size=500, 
                                 alpha=0.8, label=subcat)
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')
        
        plt.title(f"Dark Souls Knowledge Graph - {category} Category", size=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        
        filename = f"knowledge_graph/{category.replace(' ', '_').lower()}_graph.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 {category} graph saved to {filename}")
        plt.show()
    
    def create_dialogue_graph(self) -> None:
        """Create a graph visualization focusing on dialogue relationships"""
        # Filter for likely dialogue-related entities
        dialogue_keywords = ['says', 'speaks', 'tells', 'asks', 'dialogue', 'voice', 'words']
        dialogue_rels = self.relationships_df[
            self.relationships_df['predicate'].str.contains('|'.join(dialogue_keywords), case=False, na=False)
        ]
        
        # Get entities involved in dialogue
        dialogue_entities = set(dialogue_rels['subject'].tolist() + dialogue_rels['object'].tolist())
        
        if not dialogue_entities:
            print("⚠️ No dialogue relationships found")
            return
        
        # Create subgraph
        subgraph = self.graph.subgraph(dialogue_entities)
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Color nodes by entity type
        entity_types = {}
        for node in subgraph.nodes():
            entity_row = self.entities_df[self.entities_df['entity'] == node]
            if not entity_row.empty:
                entity_types[node] = entity_row.iloc[0]['type']
            else:
                entity_types[node] = 'UNKNOWN'
        
        type_colors = {'PERSON': 'lightblue', 'ORG': 'lightgreen', 'LOC': 'lightcoral', 
                      'GPE': 'lightyellow', 'WORK_OF_ART': 'lightpink', 'UNKNOWN': 'lightgray'}
        
        for entity_type, color in type_colors.items():
            nodes_of_type = [node for node, ntype in entity_types.items() if ntype == entity_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes_of_type, 
                                     node_color=color, node_size=600, alpha=0.8, label=entity_type)
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.6, edge_color='darkblue')
        nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold')
        
        plt.title("Dark Souls Dialogue Network", size=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig("knowledge_graph/dialogue_network.png", dpi=300, bbox_inches='tight')
        print("📊 Dialogue network saved to knowledge_graph/dialogue_network.png")
        plt.show()
    
    def create_item_relationship_graph(self) -> None:
        """Create a graph visualization focusing on item relationships"""
        # Filter for item-related relationships
        item_keywords = ['has_item', 'sword', 'weapon', 'armor', 'ring', 'shield', 'bow', 'spell']
        item_rels = self.relationships_df[
            (self.relationships_df['type'] == 'item') |
            (self.relationships_df['predicate'].str.contains('|'.join(item_keywords), case=False, na=False))
        ]
        
        # Get entities involved with items
        item_entities = set(item_rels['subject'].tolist() + item_rels['object'].tolist())
        
        if not item_entities:
            print("⚠️ No item relationships found")
            return
        
        # Create subgraph
        subgraph = self.graph.subgraph(item_entities)
        
        plt.figure(figsize=(18, 14))
        pos = nx.spring_layout(subgraph, k=2.5, iterations=50)
        
        # Color nodes by category
        entity_categories = {}
        for node in subgraph.nodes():
            entity_row = self.entities_df[self.entities_df['entity'] == node]
            if not entity_row.empty:
                entity_categories[node] = entity_row.iloc[0]['category']
            else:
                entity_categories[node] = 'Unknown'
        
        categories = list(set(entity_categories.values()))
        category_colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, category_colors))
        
        for category in categories:
            nodes_of_category = [node for node, cat in entity_categories.items() if cat == category]
            if nodes_of_category:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes_of_category, 
                                     node_color=[color_map[category]], node_size=400, 
                                     alpha=0.8, label=category)
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.4, edge_color='gray')
        nx.draw_networkx_labels(subgraph, pos, font_size=7, font_weight='bold')
        
        plt.title("Dark Souls Item Relationship Network", size=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig("knowledge_graph/item_network.png", dpi=300, bbox_inches='tight')
        print("📊 Item network saved to knowledge_graph/item_network.png")
        plt.show()
    
    def analyze_category_statistics(self, category: str) -> Dict[str, Any]:
        """Analyze statistics for a specific category"""
        category_entities = self.entities_df[self.entities_df['category'] == category]
        category_rels = self.relationships_df[self.relationships_df['category'] == category]
        
        # Get subcategory breakdown
        subcategory_counts = category_entities['subcategory'].value_counts().to_dict()
        
        # Get relationship types
        rel_type_counts = category_rels['type'].value_counts().to_dict()
        
        # Get most connected entities in this category
        category_nodes = category_entities['entity'].tolist()
        subgraph = self.graph.subgraph(category_nodes)
        degree_centrality = nx.degree_centrality(subgraph)
        top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "category": category,
            "total_entities": len(category_entities),
            "total_relationships": len(category_rels),
            "subcategory_breakdown": subcategory_counts,
            "relationship_types": rel_type_counts,
            "most_connected_entities": top_connected,
            "graph_density": nx.density(subgraph),
            "connected_components": nx.number_connected_components(subgraph)
        }
    
    def generate_cypher_queries(self) -> Dict[str, str]:
        """Generate Cypher queries for Neo4j graph database"""
        queries = {
            "create_entities": """
// Create entity nodes
LOAD CSV WITH HEADERS FROM 'file:///entities.csv' AS row
CREATE (e:Entity {
    name: row.entity,
    type: row.type,
    category: row.category,
    subcategory: row.subcategory,
    source_url: row.source_url
})
""",
            
            "create_relationships": """
// Create relationships
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (s:Entity {name: row.subject})
MATCH (o:Entity {name: row.object})
CREATE (s)-[r:RELATES_TO {
    predicate: row.predicate,
    type: row.type,
    category: row.category,
    subcategory: row.subcategory,
    source_url: row.source_url
}]->(o)
""",
            
            "find_most_connected": """
// Find most connected entities
MATCH (e:Entity)
RETURN e.name, e.type, e.category, 
       size((e)-[]-()) as connections
ORDER BY connections DESC
LIMIT 20
""",
            
            "weapon_relationships": """
// Find all weapon relationships
MATCH (e:Entity)-[r]->(w:Entity)
WHERE e.category = 'Weapons' OR w.category = 'Weapons'
RETURN e.name, type(r), r.predicate, w.name, e.category, w.category
ORDER BY e.name
""",
            
            "character_dialogues": """
// Find character dialogue patterns
MATCH (p:Entity {type: 'PERSON'})-[r]->(target)
WHERE r.predicate CONTAINS 'say' OR r.predicate CONTAINS 'tell' 
   OR r.predicate CONTAINS 'speak' OR r.predicate CONTAINS 'ask'
RETURN p.name as character, r.predicate as dialogue_type, 
       target.name as target, target.type
ORDER BY p.name
""",
            
            "item_ownership": """
// Find item ownership patterns
MATCH (owner)-[r {type: 'item'}]->(item)
RETURN owner.name as owner, owner.type as owner_type,
       r.predicate as relationship, item.name as item, 
       item.category as item_category
ORDER BY owner.name
""",
            
            "location_connections": """
// Find location-based connections
MATCH (l:Entity {type: 'LOC'})-[r]-(connected)
RETURN l.name as location, l.category,
       connected.name as connected_entity, connected.type,
       r.predicate as relationship_type
ORDER BY l.name
""",
            
            "category_analysis": """
// Analyze relationships within categories
MATCH (e1:Entity)-[r]->(e2:Entity)
WHERE e1.category = e2.category
RETURN e1.category as category, 
       count(r) as internal_relationships,
       collect(DISTINCT r.type) as relationship_types
ORDER BY internal_relationships DESC
""",
            
            "shortest_path_entities": """
// Find shortest path between two specific entities
MATCH (start:Entity {name: 'Gwyn'}), (end:Entity {name: 'Artorias'}),
      path = shortestPath((start)-[*]-(end))
RETURN [node in nodes(path) | node.name] as path_entities,
       [rel in relationships(path) | rel.predicate] as relationships
""",
            
            "faction_analysis": """
// Analyze faction relationships
MATCH (e:Entity)-[r {type: 'faction'}]->(f)
RETURN f.name as faction, count(e) as members,
       collect(e.name) as member_list
ORDER BY members DESC
""",
            
            "community_detection": """
// Detect communities using label propagation
CALL gds.labelPropagation.stream('knowledge_graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name as entity, communityId
ORDER BY communityId, entity
""",
            
            "centrality_analysis": """
// Calculate various centrality measures
CALL gds.pageRank.stream('knowledge_graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name as entity, score as pagerank_score
ORDER BY score DESC
LIMIT 20
"""
        }
        
        return queries

from pydantic import BaseModel, Field
from typing import ClassVar, Any

class KnowledgeGraphTool(BaseTool):
    name: str = "knowledge_graph_tool"
    description: str = "Query the Dark Souls knowledge graph for entities, relationships, and analysis"
    analyzer: Any = Field(default=None, exclude=True)
    
    def __init__(self, analyzer: KnowledgeGraphAnalyzer):
        super().__init__()
        self.analyzer = analyzer
    
    def _run(self, query: str) -> str:
        """Execute knowledge graph query"""
        query = query.lower()
        
        if "centrality" in query or "important" in query:
            centrality = self.analyzer.get_entity_centrality()
            top_entities = sorted(centrality['betweenness'].items(), key=lambda x: x[1], reverse=True)[:10]
            return f"Most central entities: {top_entities}"
        
        elif "path between" in query or "connection" in query:
            # Extract entities from query (simplified)
            entities = [word.title() for word in query.split() if word.title() in self.analyzer.graph.nodes]
            if len(entities) >= 2:
                path = self.analyzer.find_shortest_path(entities[0], entities[1])
                return f"Path between {entities[0]} and {entities[1]}: {path}"
        
        elif "analyze" in query:
            entities = [word.title() for word in query.split() if word.title() in self.analyzer.graph.nodes]
            if entities:
                analysis = self.analyzer.analyze_entity_relationships(entities[0])
                return f"Analysis for {entities[0]}: {json.dumps(analysis, indent=2)}"
        
        elif "statistics" in query or "overview" in query:
            stats = self.analyzer.get_graph_statistics()
            return f"Graph Statistics: {json.dumps(stats, indent=2)}"
        
        elif "communities" in query:
            communities = self.analyzer.detect_communities()
            return f"Found {len(communities)} communities. Largest has {len(max(communities, key=len)) if communities else 0} members."
        
        return "Query not understood. Try asking about centrality, paths between entities, entity analysis, graph statistics, or communities."
    
    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")

# NEW: Enhanced Dark Souls QA System
class DarkSoulsQASystem:
    def __init__(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame, analyzer: KnowledgeGraphAnalyzer = None, openai_api_key: str = None):
        self.entities_df = entities_df
        self.relationships_df = relationships_df
        # Reuse analyzer if provided to avoid rebuilding graph
        self.analyzer = analyzer or KnowledgeGraphAnalyzer(entities_df, relationships_df)
        
        # Set OpenAI API key
        api_key = openai_api_key or "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"
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-3.5-turbo",
                openai_api_key=api_key
            )
            print("✅ OpenAI LLM initialized successfully!")
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI LLM: {e}")
            self.llm = None
        
        # Initialize memory and tools
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.kg_tool = KnowledgeGraphTool(self.analyzer)
        
        # Initialize agent if LLM is available
        if self.llm:
            self.agent = initialize_agent(
                tools=[self.kg_tool],
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True
            )
    
    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Search for entities matching the query with fuzzy matching"""
        query_lower = query.lower()
        matches = []
        exact_matches = []
        partial_matches = []
        
        for _, entity in self.entities_df.iterrows():
            entity_name = entity['entity'].lower()
            if entity_name == query_lower:
                exact_matches.append({
                    'entity': entity['entity'],
                    'type': entity['type'],
                    'category': entity['category'],
                    'subcategory': entity['subcategory'],
                    'match_type': 'exact'
                })
            elif query_lower in entity_name:
                partial_matches.append({
                    'entity': entity['entity'],
                    'type': entity['type'],
                    'category': entity['category'],
                    'subcategory': entity['subcategory'],
                    'match_type': 'partial'
                })
        
        # Return exact matches first, then partial matches
        return exact_matches + partial_matches[:15]
    
    def find_characters(self, query: str = "") -> List[Dict[str, Any]]:
        """Find character entities (PERSON type)"""
        characters = self.entities_df[self.entities_df['type'] == 'PERSON']
        
        if query:
            query_lower = query.lower()
            characters = characters[characters['entity'].str.lower().str.contains(query_lower, na=False)]
        
        return characters.to_dict('records')
    
    def find_relationships_by_type(self, rel_type: str) -> List[Dict[str, Any]]:
        """Find relationships by type (lore, item, faction, etc.)"""
        filtered_rels = self.relationships_df[self.relationships_df['type'] == rel_type]
        return filtered_rels.to_dict('records')
    
    def get_character_story(self, character_name: str) -> Dict[str, Any]:
        """Get comprehensive story information for a character"""
        # Find character
        character_matches = self.entities_df[
            self.entities_df['entity'].str.contains(character_name, case=False, na=False) &
            (self.entities_df['type'] == 'PERSON')
        ]
        
        if character_matches.empty:
            return {"error": f"Character '{character_name}' not found"}
        
        character = character_matches.iloc[0]['entity']
        
        # Get all relationships involving this character
        char_relationships = self.relationships_df[
            (self.relationships_df['subject'].str.contains(character, case=False, na=False)) |
            (self.relationships_df['object'].str.contains(character, case=False, na=False))
        ]
        
        # Categorize relationships
        lore_rels = char_relationships[char_relationships['type'] == 'lore']
        item_rels = char_relationships[char_relationships['type'] == 'item']
        faction_rels = char_relationships[char_relationships['type'] == 'faction']
        
        # Get connected characters
        connected_chars = set()
        for _, rel in char_relationships.iterrows():
            if rel['subject'] != character:
                connected_chars.add(rel['subject'])
            if rel['object'] != character:
                connected_chars.add(rel['object'])
        
        connected_characters = list(connected_chars)[:10]
        
        return {
            "character": character,
            "total_relationships": len(char_relationships),
            "lore_relationships": lore_rels.to_dict('records'),
            "item_relationships": item_rels.to_dict('records'),
            "faction_relationships": faction_rels.to_dict('records'),
            "connected_characters": connected_characters,
            "character_info": character_matches.iloc[0].to_dict()
        }
    
    def get_relationship_network(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get detailed relationship network for an entity"""
        if entity not in self.analyzer.graph.nodes:
            return {"error": f"Entity '{entity}' not found in graph"}
        
        network = {
            "center_entity": entity,
            "relationships_by_depth": {},
            "relationship_types": {},
            "connected_entities": {}
        }
        
        for depth in range(1, max_depth + 1):
            # Get entities at this depth
            entities_at_depth = []
            for node in self.analyzer.graph.nodes():
                try:
                    path_length = nx.shortest_path_length(self.analyzer.graph, entity, node)
                    if path_length == depth:
                        entities_at_depth.append(node)
                except nx.NetworkXNoPath:
                    continue
            
            network["relationships_by_depth"][f"depth_{depth}"] = entities_at_depth
            
            # Get relationship details for this depth
            if depth == 1:  # Direct relationships only
                direct_rels = self.relationships_df[
                    (self.relationships_df['subject'] == entity) |
                    (self.relationships_df['object'] == entity)
                ]
                
                rel_types = {}
                for _, rel in direct_rels.iterrows():
                    rel_type = rel['type']
                    if rel_type not in rel_types:
                        rel_types[rel_type] = []
                    rel_types[rel_type].append({
                        'subject': rel['subject'],
                        'predicate': rel['predicate'],
                        'object': rel['object']
                    })
                
                network["relationship_types"] = rel_types
        
        return network
    
    def get_entity_context(self, entity_name: str) -> str:
        """Get contextual information about an entity"""
        # Get entity info
        entity_info = self.entities_df[self.entities_df['entity'].str.contains(entity_name, case=False, na=False)]
        
        # Get relationships
        relationships = self.relationships_df[
            (self.relationships_df['subject'].str.contains(entity_name, case=False, na=False)) |
            (self.relationships_df['object'].str.contains(entity_name, case=False, na=False))
        ]
        
        context = f"Entity: {entity_name}\n"
        if not entity_info.empty:
            context += f"Type: {entity_info.iloc[0]['type']}\n"
            context += f"Category: {entity_info.iloc[0]['category']}\n"
            context += f"Subcategory: {entity_info.iloc[0]['subcategory']}\n"
        
        context += "\nRelationships:\n"
        for _, rel in relationships.head(10).iterrows():
            context += f"- {rel['subject']} {rel['predicate']} {rel['object']} ({rel['type']})\n"
        
        return context
    
    def answer_question(self, question: str) -> str:
        """Answer a question about Dark Souls using the knowledge graph"""
        if not self.llm:
            return "LLM not available. Please provide an OpenAI API key."
        
        # Pre-process question to extract entities and provide context
        question_lower = question.lower()
        context_info = ""
        
        # Try to identify entities mentioned in the question
        entities_mentioned = []
        for _, entity in self.entities_df.iterrows():
            if entity['entity'].lower() in question_lower:
                entities_mentioned.append(entity['entity'])
        
        # Get context for mentioned entities
        if entities_mentioned:
            context_info = "\n\nRelevant entities found in question:\n"
            for entity in entities_mentioned[:3]:  # Limit to 3 for context
                context = self.get_entity_context(entity)
                context_info += f"\n{context}\n"
        
        # Enhanced system prompt with more specific instructions
        system_prompt = f"""You are a Dark Souls lore expert with access to a comprehensive knowledge graph containing {len(self.entities_df)} entities and {len(self.relationships_df)} relationships.

The knowledge graph includes:
- Characters (PERSON entities): NPCs, bosses, historical figures
- Locations (LOC/GPE entities): Areas, kingdoms, specific places  
- Items (WORK_OF_ART entities): Weapons, armor, rings, consumables
- Organizations: Covenants, factions, groups

When answering questions:
1. Use the knowledge graph tool to find specific relationships and entities
2. Provide detailed, lore-accurate information
3. Cite specific connections between entities when possible
4. If asking about characters, include their relationships, items, and story connections
5. Be engaging and immersive while staying accurate
6. If information isn't in the graph, say so clearly

{context_info}

Remember: You have access to relationship types including lore, item, faction, and general connections."""
        
        try:
            # Create enhanced input with context
            enhanced_question = f"{question}\n\nPlease use the knowledge graph to provide a comprehensive answer with specific entity relationships and connections."
            response = self.agent.run(input=enhanced_question)
            return response
        except Exception as e:
            return f"Error processing question: {str(e)}\n\nTry using specific commands like 'story [character]' or 'search [entity]' for better results."
    
    def start_interactive_session(self):
        """Start an interactive Q&A session"""
        print("🎮 Welcome to the Enhanced Dark Souls Knowledge Graph Q&A System!")
        print("Ask me anything about Dark Souls lore, characters, items, or locations.")
        print("Type 'quit' to exit, 'stats' for graph statistics, or 'help' for commands.\n")
        
        while True:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye! May the flames guide thee.")
                break
            elif question.lower() == 'help':
                print("""
🎮 Enhanced Available Commands:
══════════════════════════════

📖 General Questions:
- Ask any natural language question about Dark Souls
- Examples: "Tell me about Artorias", "What weapons does Ornstein use?"

🔍 Search & Discovery:
- 'search [name]' - Search for any entity (characters, items, locations)
- 'characters' - List all characters
- 'characters [name]' - Search for specific characters
- 'story [character]' - Get complete character story and relationships

🔗 Relationship Analysis:
- 'analyze [entity]' - Deep analyze entity relationships  
- 'network [entity]' - Show relationship network
- 'path [entity1] to [entity2]' - Find connection path between entities
- 'relationships [type]' - Show relationships by type (lore, item, faction)

📊 Advanced Graph Analysis:
- 'communities' - Detect communities in the graph
- 'motifs' - Analyze graph motifs and patterns
- 'centrality [entity]' - Get centrality measures for entity
- 'category [name]' - Analyze specific category statistics
- 'stats' - Show comprehensive graph statistics

🎨 Enhanced Visualizations:
- 'interactive [category]' - Create interactive graph for category
- 'graphs' - Create all graph visualizations
- 'dialogues' - Analyze dialogue patterns and create network
- 'items' - Analyze item relationships and create network

📦 Data Export:
- 'export' - Create comprehensive data export (CSV, Excel, ZIP)
- 'download' - Same as export

💾 Database Queries:
- 'cypher' - Show Neo4j Cypher query examples
- 'neo4j' - Advanced graph database queries

⚡ Quick Tips:
- Use 'interactive' for better visualizations
- Try 'communities' to find character groups
- Use 'export' to download all data
- 'motifs' shows interesting graph patterns

Type 'quit' to exit the system
                """)
                continue
            elif question.lower() == 'stats':
                stats = self.analyzer.get_graph_statistics()
                print(f"\n📊 Comprehensive Knowledge Graph Statistics:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
                continue
            elif question.lower() in ['export', 'download']:
                print("\n📦 Creating comprehensive export package...")
                export_dir, zip_file = create_comprehensive_export()
                if export_dir:
                    print("✅ Export completed successfully!")
                continue
            elif question.lower() == 'communities':
                print("\n🏘️ Detecting communities in the knowledge graph...")
                communities = self.analyzer.detect_communities()
                print(f"Found {len(communities)} communities:")
                for i, community in enumerate(communities[:10], 1):  # Show first 10
                    print(f"  Community {i}: {len(community)} members")
                    print(f"    Members: {', '.join(community[:5])}" + 
                          (f" (+{len(community)-5} more)" if len(community) > 5 else ""))
                continue
            elif question.lower() == 'motifs':
                print("\n🔍 Analyzing graph motifs and patterns...")
                motifs = self.analyzer.analyze_graph_motifs()
                print("Graph Motifs Analysis:")
                for key, value in motifs.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} found")
                        if value:
                            print(f"    Examples: {value[:3]}")
                    else:
                        print(f"  {key}: {value}")
                continue
            elif question.lower().startswith('interactive'):
                parts = question.split()
                category = parts[1] if len(parts) > 1 else None
                print(f"\n🎨 Creating interactive visualization{' for ' + category if category else ''}...")
                self.analyzer.create_interactive_graph(category)
                continue
            elif question.lower().startswith('centrality '):
                entity = question[11:].strip()
                centrality_data = self.analyzer.get_entity_centrality()
                print(f"\n📊 Centrality measures for '{entity}':")
                for measure, values in centrality_data.items():
                    score = values.get(entity, 0)
                    print(f"  {measure}: {score:.4f}")
                continue
            elif question.lower().startswith('search '):
                entity = question[7:]
                matches = self.search_entities(entity)
                print(f"\n🔍 Found {len(matches)} matches for '{entity}':")
                for match in matches:
                    match_type = match.get('match_type', 'partial')
                    print(f"  {'★' if match_type == 'exact' else '-'} {match['entity']} ({match['type']}, {match['category']})")
                continue
            elif question.lower() == 'characters':
                characters = self.find_characters()
                print(f"\n👥 Found {len(characters)} characters:")
                for char in characters[:20]:  # Show first 20
                    print(f"  - {char['entity']} ({char['category']})")
                if len(characters) > 20:
                    print(f"  ... and {len(characters) - 20} more. Use 'search [name]' for specific characters.")
                continue
            elif question.lower().startswith('characters '):
                char_query = question[11:]
                characters = self.find_characters(char_query)
                print(f"\n👥 Found {len(characters)} characters matching '{char_query}':")
                for char in characters:
                    print(f"  - {char['entity']} ({char['category']}, {char['subcategory']})")
                continue
            elif question.lower().startswith('story '):
                character = question[6:].strip()
                story = self.get_character_story(character)
                if "error" in story:
                    print(f"\n❌ {story['error']}")
                    # Suggest similar characters
                    similar = self.find_characters(character)
                    if similar:
                        print("🔍 Did you mean one of these characters?")
                        for char in similar[:5]:
                            print(f"  - {char['entity']}")
                else:
                    print(f"\n📜 Story of {story['character']}:")
                    print(f"📊 Total Relationships: {story['total_relationships']}")
                    
                    if story['lore_relationships']:
                        print(f"\n🏛️ Lore Relationships ({len(story['lore_relationships'])}):")
                        for rel in story['lore_relationships'][:5]:
                            print(f"  - {rel['subject']} {rel['predicate']} {rel['object']}")
                        if len(story['lore_relationships']) > 5:
                            print(f"  ... and {len(story['lore_relationships']) - 5} more")
                    
                    if story['item_relationships']:
                        print(f"\n⚔️ Item Relationships ({len(story['item_relationships'])}):")
                        for rel in story['item_relationships'][:5]:
                            print(f"  - {rel['subject']} {rel['predicate']} {rel['object']}")
                    
                    if story['faction_relationships']:
                        print(f"\n🛡️ Faction Relationships ({len(story['faction_relationships'])}):")
                        for rel in story['faction_relationships']:
                            print(f"  - {rel['subject']} {rel['predicate']} {rel['object']}")
                    
                    if story['connected_characters']:
                        print(f"\n👥 Connected Characters:")
                        for char in story['connected_characters']:
                            print(f"  - {char}")
                continue
            elif question.lower().startswith('network '):
                entity = question[8:].strip()
                network = self.get_relationship_network(entity)
                if "error" in network:
                    print(f"\n❌ {network['error']}")
                else:
                    print(f"\n🕸️ Relationship Network for {network['center_entity']}:")
                    for depth, entities in network['relationships_by_depth'].items():
                        if entities:
                            print(f"\n{depth.replace('_', ' ').title()}: {len(entities)} entities")
                            for entity_name in entities[:10]:
                                print(f"  - {entity_name}")
                            if len(entities) > 10:
                                print(f"  ... and {len(entities) - 10} more")
                    
                    if network['relationship_types']:
                        print(f"\n🔗 Direct Relationship Types:")
                        for rel_type, rels in network['relationship_types'].items():
                            print(f"  {rel_type}: {len(rels)} relationships")
                continue
            elif question.lower().startswith('relationships '):
                rel_type = question[14:].strip()
                relationships = self.find_relationships_by_type(rel_type)
                print(f"\n🔗 Found {len(relationships)} '{rel_type}' relationships:")
                for rel in relationships[:15]:
                    print(f"  - {rel['subject']} {rel['predicate']} {rel['object']}")
                if len(relationships) > 15:
                    print(f"  ... and {len(relationships) - 15} more")
                
                # Show available relationship types
                all_types = self.relationships_df['type'].value_counts()
                print(f"\n📋 Available relationship types:")
                for rtype, count in all_types.items():
                    marker = "★" if rtype == rel_type else "-"
                    print(f"  {marker} {rtype}: {count} relationships")
                continue
            elif question.lower().startswith('analyze '):
                entity = question[8:].title()
                analysis = self.analyzer.analyze_entity_relationships(entity)
                if "error" in analysis:
                    print(f"\n❌ {analysis['error']}")
                else:
                    print(f"\n📈 Enhanced Analysis for {entity}:")
                    print(f"  Degree: {analysis['degree']}")
                    print(f"  Clustering: {analysis['clustering_coefficient']:.3f}")
                    print(f"  Direct relationships: {len(analysis['direct_relationships'])}")
                    print(f"  Centrality measures:")
                    for measure, score in analysis['centrality_measures'].items():
                        print(f"    {measure}: {score:.4f}")
                    print(f"  Relationship types: {analysis['relationship_types']}")
                continue
            elif "path" in question.lower() and " to " in question.lower():
                parts = question.lower().replace("path", "").replace("to", "|").strip().split("|")
                if len(parts) == 2:
                    entity1, entity2 = parts[0].strip().title(), parts[1].strip().title()
                    path = self.analyzer.find_shortest_path(entity1, entity2)
                    if path:
                        print(f"\n🛤️ Path from {entity1} to {entity2}: {' -> '.join(path)}")
                    else:
                        print(f"\n❌ No path found between {entity1} and {entity2}")
                continue
            elif question.lower().startswith('category '):
                category = question[9:].strip()
                categories = self.entities_df['category'].unique()
                matching_cats = [cat for cat in categories if category.lower() in cat.lower()]
                if matching_cats:
                    analysis = self.analyzer.analyze_category_statistics(matching_cats[0])
                    print(f"\n📊 Analysis for {matching_cats[0]} category:")
                    for key, value in analysis.items():
                        if key == 'most_connected_entities':
                            print(f"  {key}: {value[:5]}")  # Show top 5
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"\n❌ Category '{category}' not found. Available categories:")
                    for cat in sorted(categories):
                        print(f"    - {cat}")
                continue
            elif question.lower() == 'graphs':
                print("\n🎨 Creating all graph visualizations...")
                create_all_visualizations(self.analyzer)
                continue
            elif question.lower() in ['cypher', 'neo4j']:
                queries = self.analyzer.generate_cypher_queries()
                print("\n🔍 Neo4j Cypher Query Examples:")
                for name, query in queries.items():
                    print(f"\n--- {name.replace('_', ' ').title()} ---")
                    print(query)
                continue
            elif question.lower() == 'dialogues':
                print("\n🗣️ Creating dialogue network analysis...")
                self.analyzer.create_dialogue_graph()
                
                # Show dialogue statistics
                dialogue_keywords = ['says', 'speaks', 'tells', 'asks', 'dialogue', 'voice', 'words']
                dialogue_rels = self.analyzer.relationships_df[
                    self.analyzer.relationships_df['predicate'].str.contains('|'.join(dialogue_keywords), case=False, na=False)
                ]
                print(f"Found {len(dialogue_rels)} dialogue-related relationships")
                if len(dialogue_rels) > 0:
                    print("Top dialogue predicates:")
                    for pred, count in dialogue_rels['predicate'].value_counts().head(10).items():
                        print(f"  {pred}: {count}")
                continue
            elif question.lower() == 'items':
                print("\n⚔️ Creating item relationship analysis...")
                self.analyzer.create_item_relationship_graph()
                
                # Show item statistics
                item_rels = self.analyzer.relationships_df[self.analyzer.relationships_df['type'] == 'item']
                print(f"Found {len(item_rels)} item-related relationships")
                if len(item_rels) > 0:
                    print("Top item relationship types:")
                    for pred, count in item_rels['predicate'].value_counts().head(10).items():
                        print(f"  {pred}: {count}")
                continue
            
            if not question:
                continue
            
            print("\n🤖 Analyzing knowledge graph...")
            answer = self.answer_question(question)
            print(f"\n📖 Answer: {answer}")

def load_existing_data():
    """Load existing knowledge graph data with optimization"""
    try:
        print("📂 Loading knowledge graph data...")
        
        # Use efficient data types and chunking for large files
        entities_df = pd.read_csv("knowledge_graph/entities.csv", 
                                dtype={'entity': 'string', 'type': 'category', 
                                      'category': 'category', 'subcategory': 'category'})
        relationships_df = pd.read_csv("knowledge_graph/relationships.csv",
                                     dtype={'subject': 'string', 'object': 'string',
                                           'predicate': 'string', 'type': 'category',
                                           'category': 'category', 'subcategory': 'category'})
        
        # Remove duplicates to speed up processing
        entities_df = entities_df.drop_duplicates(subset=['entity'], keep='first')
        relationships_df = relationships_df.drop_duplicates(keep='first')
        
        print(f"✅ Loaded: {len(entities_df)} entities, {len(relationships_df)} relationships")
        return entities_df, relationships_df
    except FileNotFoundError:
        print("❌ No existing knowledge graph found. Please run data collection first.")
        return None, None

def visualize_graph_sample(analyzer: KnowledgeGraphAnalyzer, max_nodes: int = 50):
    """Visualize a sample of the knowledge graph with optimized rendering"""
    try:
        # Skip visualization if file already exists
        viz_path = "knowledge_graph/graph_visualization.png"
        if os.path.exists(viz_path):
            print("📊 Graph visualization already exists, skipping regeneration")
            return
            
        print("🎨 Creating graph visualization...")
        # Get a subgraph with most connected nodes
        centrality = analyzer.get_entity_centrality()['degree']
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node[0] for node in top_nodes]
        
        subgraph = analyzer.graph.subgraph(top_node_names)
        
        plt.figure(figsize=(15, 12))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes with different colors based on entity type
        entity_types = {}
        for node in subgraph.nodes():
            entity_row = analyzer.entities_df[analyzer.entities_df['entity'] == node]
            if not entity_row.empty:
                entity_types[node] = entity_row.iloc[0]['type']
            else:
                entity_types[node] = 'UNKNOWN'
        
        type_colors = {'PERSON': 'lightblue', 'ORG': 'lightgreen', 'LOC': 'lightcoral', 
                      'GPE': 'lightyellow', 'WORK_OF_ART': 'lightpink', 'UNKNOWN': 'lightgray'}
        
        for entity_type, color in type_colors.items():
            nodes_of_type = [node for node, ntype in entity_types.items() if ntype == entity_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes_of_type, 
                                     node_color=color, node_size=300, alpha=0.8)
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')
        
        plt.title("Dark Souls Knowledge Graph (Sample)", size=16)
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=etype) 
                           for etype, color in type_colors.items()], loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig("knowledge_graph/graph_visualization.png", dpi=300, bbox_inches='tight')
        print("📊 Graph visualization saved to knowledge_graph/graph_visualization.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def create_all_visualizations(analyzer: KnowledgeGraphAnalyzer):
    """Create all different types of graph visualizations"""
    import numpy as np
    
    print("🎨 Creating comprehensive graph visualizations...")
    
    # 1. Overall sample graph
    visualize_graph_sample(analyzer)
    
    # 2. Interactive graph for top categories
    categories = analyzer.entities_df['category'].value_counts().head(3).index.tolist()
    print(f"\n📊 Creating interactive graphs for top categories...")
    for category in categories:
        print(f"  📈 Creating interactive graph for {category}")
        analyzer.create_interactive_graph(category)
    
    # 3. Dialogue network
    print("\n🗣️ Creating dialogue network...")
    analyzer.create_dialogue_graph()
    
    # 4. Item relationship network
    print("\n⚔️ Creating item relationship network...")
    analyzer.create_item_relationship_graph()
    
    print("\n✅ All visualizations created successfully!")

def main():
    """Main function to run the Enhanced Dark Souls Knowledge Graph system"""
    start_time = time.time()
    print("🎮 Enhanced Dark Souls Knowledge Graph System")
    print("=" * 50)
    print("🚀 Performance Optimizations Active")
    print("📊 Advanced NLP Features Enabled")
    print("📦 Export & Visualization Tools Ready")
    print("=" * 50)
    
    # Check if data exists
    load_start = time.time()
    entities_df, relationships_df = load_existing_data()
    print(f"⏱️ Data loading took {time.time() - load_start:.2f} seconds")
    
    if entities_df is None or relationships_df is None:
        print("🔄 Starting enhanced data collection process...")
        # Run the enhanced data collection and processing
        all_links_by_category = scrape_all_data()
        
        # Scrape and process data with parallel processing
        os.makedirs("darksouls_data", exist_ok=True)
        visited_urls = set()
        all_entities = []
        all_relationships = []
        all_advanced_features = []
        
        print("\n🚀 Starting parallel content scraping...")
        for category, subcats in all_links_by_category.items():
            print(f"\n📂 Processing category: {category}")
            category_data = []

            for subcat, urls in subcats.items():
                print(f"  ⬇️ Subcategory: {subcat} ({len(urls)} pages)")

                # Use ThreadPoolExecutor for faster scraping
                with ThreadPoolExecutor(max_workers=3) as executor:
                    url_content_futures = {}
                    for url in urls:
                        if url not in visited_urls:
                            visited_urls.add(url)
                            future = executor.submit(scrape_page_content, url)
                            url_content_futures[future] = url
                    
                    for future in as_completed(url_content_futures):
                        url = url_content_futures[future]
                        try:
                            content = future.result()
                            if content:
                                category_data.append({
                                    "Subcategory": subcat,
                                    "URL": url,
                                    "Content": content
                                })
                        except Exception as e:
                            print(f"    ❌ Failed to process {url}: {e}")

            if category_data:
                df = pd.DataFrame(category_data)
                safe_name = re.sub(r'[\\/*?:"<>|]', "", category)
                df.to_csv(f"darksouls_data/{safe_name}.csv", index=False)
                print(f"  💾 Saved {len(category_data)} pages to {safe_name}.csv")
        
        # Enhanced NLP processing
        print("\n🔮 Starting enhanced NLP processing...")
        for file in os.listdir("darksouls_data"):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join("darksouls_data", file)
            try:
                df = pd.read_csv(file_path)
                category = file.replace(".csv", "")

                if df.empty or 'Content' not in df.columns:
                    continue

                print(f"  📄 Processing {file} ({len(df)} pages) with advanced NLP...")

                for _, row in df.iterrows():
                    content = row.get('Content', '')
                    if not content or len(content) < 100:
                        continue

                    subcat = row.get('Subcategory', '')
                    url = row.get('URL', '')

                    # Extract all information with enhanced features
                    entities = extract_entities(content)
                    relationships_list = extract_relationships(content)
                    lore_relationships = extract_lore_relationships(content)
                    item_lore = extract_item_lore(content)
                    faction_rels = extract_faction_relationships(content)
                    
                    # Advanced NLP features
                    sentiment = extract_sentiment(content)
                    temporal_exprs = extract_temporal_expressions(content)
                    char_attributes = extract_character_attributes(content)
                    co_occurrences = extract_co_occurrence_patterns(content)
                    hierarchical_rels = extract_hierarchical_relationships(content)

                    # Store entities with sentiment
                    for ent, label in entities:
                        all_entities.append({
                            "entity": ent,
                            "type": label,
                            "source_url": url,
                            "category": category,
                            "subcategory": subcat,
                            "sentiment": sentiment
                        })

                    # Store all relationship types
                    for rel in relationships_list:
                        all_relationships.append({
                            "subject": rel[0],
                            "predicate": rel[1],
                            "object": rel[2],
                            "type": "general",
                            "source_url": url,
                            "category": category,
                            "subcategory": subcat
                        })

                    for rel in lore_relationships:
                        all_relationships.append({
                            "subject": rel[0],
                            "predicate": rel[1],
                            "object": rel[2],
                            "type": "lore",
                            "source_url": url,
                            "category": category,
                            "subcategory": subcat
                        })

                    for rel in item_lore:
                        all_relationships.append({
                            "subject": rel[0],
                            "predicate": rel[1],
                            "object": rel[2],
                            "type": "item",
                            "source_url": url,
                            "category": category,
                            "subcategory": subcat
                        })

                    for rel in faction_rels:
                        all_relationships.append({
                            "subject": rel[0],
                            "predicate": rel[1],
                            "object": rel[2],
                            "type": "faction",
                            "source_url": url,
                            "category": category,
                            "subcategory": subcat
                        })
                    
                    for rel in hierarchical_rels:
                        all_relationships.append({
                            "subject": str(rel.get('entities', [])),
                            "predicate": f"{rel['superior_role']}_to_{rel['subordinate_role']}",
                            "object": rel['context'][:100],
                            "type": "hierarchical",
                            "source_url": url,
                            "category": category,
                            "subcategory": subcat
                        })
                    
                    # Store advanced features
                    all_advanced_features.extend([
                        {"type": "temporal", "data": temporal_exprs, "source_url": url, "category": category},
                        {"type": "attributes", "data": char_attributes, "source_url": url, "category": category},
                        {"type": "co_occurrence", "data": co_occurrences, "source_url": url, "category": category}
                    ])

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
        
        # Save results
        print("\n💾 Saving enhanced results...")
        entities_df = pd.DataFrame(all_entities)
        relationships_df = pd.DataFrame(all_relationships)

        os.makedirs("knowledge_graph", exist_ok=True)
        entities_df.to_csv("knowledge_graph/entities.csv", index=False)
        relationships_df.to_csv("knowledge_graph/relationships.csv", index=False)
        
        # Save advanced features
        with open("knowledge_graph/advanced_features.json", 'w') as f:
            json.dump(all_advanced_features, f, indent=2)

        print("✅ Enhanced data collection complete!")
    
    print(f"📊 Loaded: {len(entities_df)} entities, {len(relationships_df)} relationships")
    
    # Initialize enhanced analyzer and QA system
    print("\n🧠 Initializing Enhanced Knowledge Graph Analyzer...")
    analyzer = KnowledgeGraphAnalyzer(entities_df, relationships_df)
    
    print("🤖 Initializing Enhanced Q&A System...")
    qa_system = DarkSoulsQASystem(entities_df, relationships_df, analyzer)
    
    # Show comprehensive graph statistics
    print("\n📈 Comprehensive Knowledge Graph Statistics:")
    stats = analyzer.get_graph_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Create visualization only if needed
    print("\n🎨 Creating enhanced graph visualization...")
    visualize_graph_sample(analyzer)
    
    # Show available categories for analysis
    print(f"\n📊 Available categories for analysis:")
    categories = entities_df['category'].unique()
    for i, cat in enumerate(sorted(categories), 1):
        entity_count = len(entities_df[entities_df['category'] == cat])
        print(f"  {i}. {cat} ({entity_count} entities)")
    
    # Show relationship type breakdown
    print(f"\n📈 Relationship Type Breakdown:")
    rel_types = relationships_df['type'].value_counts()
    for rel_type, count in rel_types.head(10).items():
        print(f"  {rel_type}: {count} relationships")
    
    # Show most connected entities
    print("\n⭐ Most Connected Entities (by degree):")
    try:
        degree_dict = dict(analyzer.graph.degree())
        top_entities = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (entity, degree) in enumerate(top_entities, 1):
            print(f"  {i}. {entity} (connections: {degree})")
    except Exception as e:
        print(f"  Skipping entity analysis: {e}")
    
    # Show communities
    print("\n🏘️ Community Detection:")
    try:
        communities = analyzer.detect_communities()
        print(f"  Found {len(communities)} communities")
        largest_community = max(communities, key=len) if communities else []
        print(f"  Largest community: {len(largest_community)} members")
    except Exception as e:
        print(f"  Community detection: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⚡ Enhanced system initialized in {total_time:.2f} seconds")
    print("💡 Advanced optimizations active:")
    print("   - Parallel data processing")
    print("   - Enhanced NLP features")
    print("   - Graph caching enabled")
    print("   - Community detection")
    print("   - Advanced visualizations")
    print("   - Comprehensive export tools")
    
    # Auto-create exports directory
    os.makedirs("exports", exist_ok=True)
    
    print(f"\n🎯 Ready for enhanced analysis!")
    print("Type 'help' for full command list or 'export' to download all data")
    
    # Start interactive session
    qa_system.start_interactive_session()

if __name__ == "__main__":
    main()
