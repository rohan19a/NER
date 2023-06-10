import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("displaCy uses JavaScript, SVG and CSS.")
spacy.displacy.serve(doc, style="dep")

