import requests
import spacy
import nltk
from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Fetch news article from News API
def fetch_news_article(api_key, url):
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['articles'][0]['content']  # Adjust based on the API response structure
    else:
        return None

# Extract entities using SpaCy
def extract_entities_spacy(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

# Extract entities using NLTK
def extract_entities_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    chunked_tokens = ne_chunk(tagged_tokens)
    
    entities = {"PERSON": [], "ORG": [], "GPE": []}
    for chunk in chunked_tokens:
        if isinstance(chunk, nltk.Tree):
            entity_type = chunk.label()
            if entity_type in entities:
                entities[entity_type].append(' '.join(word for word, tag in chunk))
    return entities

# Compare results
def compare_results(entities_spacy, entities_nltk):
    comparison = {}
    for key in entities_spacy:
        spacy_entities = set(entities_spacy[key])
        nltk_entities = set(entities_nltk[key])
        comparison[key] = {
            "SpaCy": spacy_entities,
            "NLTK": nltk_entities,
            "Common": spacy_entities & nltk_entities,
            "Unique to SpaCy": spacy_entities - nltk_entities,
            "Unique to NLTK": nltk_entities - spacy_entities
        }
    return comparison

# Example usage
if __name__ == "__main__":
    # Replace with your News API key and URL
    API_KEY = 'c858b684138c4ac192dee23caaaeedb4'
    URL = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'
    
    article_text = fetch_news_article(API_KEY, URL)
    if article_text:
        print("Extracting entities using SpaCy...")
        entities_spacy = extract_entities_spacy(article_text)
        
        print("Extracting entities using NLTK...")
        entities_nltk = extract_entities_nltk(article_text)
        
        print("Comparing results...")
        comparison = compare_results(entities_spacy, entities_nltk)
        
        print("Entities Extracted:")
        print("SpaCy:", entities_spacy)
        print("NLTK:", entities_nltk)
        print("Comparison:", comparison)
    else:
        print("Failed to fetch news article.")
