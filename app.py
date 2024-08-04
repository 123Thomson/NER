import spacy
from spacy import displacy

# Load SpaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Large sample text for testing
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

# Process the text
doc = nlp(text)

# Print named entities
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Visualize the entities and save to an HTML file
html = displacy.render(doc, style="ent", jupyter=False)
with open("entities.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Visualization saved to entities.html. Open this file in a web browser to view the visualization.")
