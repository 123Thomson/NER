import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk

# Download required NLTK data files (only need to do this once)
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Sample contemporary news article
text = """
Apple Inc., headquartered in Cupertino, California, announced the release of its latest iPhone model on September 12, 2024. 
Tim Cook, the CEO of Apple, stated that the new iPhone has advanced features including a 48-megapixel camera and a new A17 Bionic chip. 
The announcement took place during a live event at the Steve Jobs Theater, where over 1,000 journalists and tech enthusiasts were present. 
In other news, Tesla Inc. reported a 25% increase in electric vehicle sales in Q2 2024, with a revenue of $20 billion. 
Elon Musk, CEO of Tesla, highlighted the company's commitment to expanding its battery production capabilities. 
Meanwhile, Microsoft Corporation, based in Redmond, Washington, announced a partnership with OpenAI to integrate GPT-5 into their Azure cloud services. 
Satya Nadella, CEO of Microsoft, emphasized the importance of AI in enhancing cloud computing capabilities. 
Furthermore, in the financial sector, JPMorgan Chase & Co. revealed a new investment strategy to tackle climate change, with an initial investment of $3 billion over the next five years. 
The strategy aims to reduce the carbon footprint of their investment portfolio by 40% by 2030.
"""

# Tokenize and tag parts of speech
sentences = sent_tokenize(text)
for sentence in sentences:
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    # Named Entity Chunking
    tree = ne_chunk(tagged)
    
    # Extract named entities
    nltk_entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            entity_type = subtree.label()
            nltk_entities.append((entity, entity_type))
    
    # Print results
    print("NLTK Named Entities:")
    for entity in nltk_entities:
        print(entity)
