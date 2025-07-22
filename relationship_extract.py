import os
import pandas as pd
import re
import spacy
from spacy.matcher import Matcher
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔁 Load NLP model
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# ⚔️ Lore-specific vocabulary
lore_verbs = [
    "created", "wields", "banished", "fought", "follower", "worships", "cursed",
    "defeated", "descendant", "betrayed", "guarded", "protects", "summoned", "linked",
    "resides", "haunts", "guards", "owns", "uses", "slain", "aided", "opposes"
]
item_keywords = ["sword", "ring", "armor", "pyromancy", "miracle", "talisman", "weapon", "shield", "key", "soul"]
faction_keywords = ["covenant", "faction", "order", "followers", "knights", "disciples", "legion"]
environment_keywords = ["ruins", "catacombs", "abyss", "kingdom", "fortress", "depths", "archives", "blight", "lake"]

# 🧼 Text Cleaning
def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_informative(word):
    return len(word) >= 3 and word.lower() not in stop_words and not word.isnumeric()

# 📛 Entity Filtering
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

# 👤 Named Entity Extraction
def extract_entities(text):
    doc = nlp(text)
    ents = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'WORK_OF_ART']]
    return advanced_entity_filter(ents)

# 🔗 General Relationships
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

# 📚 Lore-Specific Relationships
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

# 🛡️ Item Mentions & Associations
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

# 🏰 Environmental & Implied Lore
def extract_environment_mentions(text):
    doc = nlp(text)
    return list(set([token.text for token in doc if token.lemma_.lower() in environment_keywords]))

# 👥 Faction Relationships
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

# 🔮 Inferable Relationships (basic pronoun resolution and context)
def extract_inferable_relationships(text):
    inferable = []
    sentences = list(nlp(text).sents)
    for i in range(1, len(sentences)):
        prev_sent = sentences[i-1].text
        curr_sent = sentences[i].text
        if any(p in curr_sent for p in ["he", "she", "they", "him", "her", "his", "their"]):
            inferable.append((clean_text(prev_sent), "->", clean_text(curr_sent)))
    return inferable

# 💬 Dialogue Extractor
def extract_dialogue(text):
    dialogues = re.findall(r'["“”‘’\'](.*?)["“”‘’\']', text)
    return [dlg.strip() for dlg in dialogues if len(dlg.strip()) > 10]

# 📂 Process Dataset Folder
data_folder = "darksouls_data"

for file in os.listdir(data_folder):
    if not (file.endswith(".csv") or file.endswith(".xlsx")) or file.startswith("~$"):
        continue

    file_path = os.path.join(data_folder, file)
    try:
        df = pd.read_csv(file_path) if file.endswith(".csv") else pd.read_excel(file_path)
    except Exception as e:
        print(f"❌ Failed to read {file}: {e}")
        continue

    if 'Content' not in df.columns or df.empty:
        print(f"⚠️ Skipping {file} - Missing 'Content' or empty.")
        continue

    raw_text = df["Content"].dropna().iloc[0]
    sample_text = clean_text(raw_text)[:10000]

    # 🧠 Process the text
    entities = extract_entities(sample_text)
    relationships = extract_relationships(sample_text)
    lore_relationships = extract_lore_relationships(sample_text)
    item_mentions = extract_items(sample_text)
    item_lore = extract_item_lore(sample_text)
    faction_rels = extract_faction_relationships(sample_text)
    env_mentions = extract_environment_mentions(sample_text)
    inferred_rels = extract_inferable_relationships(sample_text)
    dialogues = extract_dialogue(sample_text)

    # 🖨️ Output
    print(f"\n📄 File: {file}")
    print(f"👤 Entities ({len(entities)}): {[ent[0] for ent in entities[:10]]}")
    print(f"🔗 General Relationships ({len(relationships)}): {relationships[:5]}")
    print(f"📚 Lore Relationships ({len(lore_relationships)}): {lore_relationships[:5]}")
    print(f"🛡️ Item Mentions ({len(item_mentions)}): {item_mentions[:5]}")
    print(f"🎯 Item Lore Relationships ({len(item_lore)}): {item_lore[:5]}")
    print(f"🏰 Environmental Mentions ({len(env_mentions)}): {env_mentions[:5]}")
    print(f"👥 Faction Relationships ({len(faction_rels)}): {faction_rels[:5]}")
    print(f"🔮 Inferable Relationships ({len(inferred_rels)}): {inferred_rels[:3]}")
    print(f"💬 Dialogues Found ({len(dialogues)}): {dialogues[:2]}")
