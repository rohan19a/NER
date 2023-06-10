#imports 
import spacy
import random
import json

def train_spacy_ner(data, iterations):
    # Load the blank English model
    nlp = spacy.blank("en")

    # Create a new entity recognizer and add it to the pipeline
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)

    # Add the annotated data to the NER training data
    for _, annotations in data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipeline components except NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # Initialize the training optimizer
        optimizer = nlp.begin_training()

        # Train the NER model
        for itn in range(iterations):
            random.shuffle(data)
            losses = {}

            for text, annotations in data:
                nlp.update(
                    [text],
                    [annotations],
                    sgd=optimizer,
                    drop=0.35,
                    losses=losses,
                )

            print(f"Iteration {itn+1}: Losses - {losses}")

    return nlp

# Example usage
if __name__ == "__main__":
    # Load your dataset in JSON format
    with open("ner_dataset.json") as file:
        dataset = json.load(file)

    # Train the NER model
    nlp = train_spacy_ner(dataset, iterations=10)

    # Save the trained model to disk
    nlp.to_disk("trained_ner_model")
