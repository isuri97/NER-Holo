import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("James was employed by IBM.He visited China during his presidency.")

relationships = {}

for ent in doc.ents:
    for ent2 in doc.ents:
        if ent.text != ent2.text:
            relationship = ""
            for token in ent.subtree:
                if token in ent2.subtree:
                    relationship = token.text
                    break
            if relationship:
                relationship_name = relationship
                if relationship_name in relationships:
                    relationships[relationship_name] += 1
                else:
                    relationships[relationship_name] = 1

for relationship, cooccurrence in relationships.items():
    print("Relationship:", relationship, "Co-occurrence:", cooccurrence)