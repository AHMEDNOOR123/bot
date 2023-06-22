import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import spacy
from pyswip import Prolog

#pytholog

def NER(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    persons = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
    return persons


model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_weights('gender_model_weights.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def predict_gender(names):
    genders = []
    for name in names:
        encoding = tokenizer([name], truncation=True, padding=True)
        input_dataset = tf.data.Dataset.from_tensor_slices(dict(encoding)).batch(1)
        predictions = model.predict(input_dataset)
        predicted_label = tf.argmax(predictions.logits, axis=1)[0].numpy()
        gender = "male" if predicted_label == 0 else "female"
        genders.append((name, gender))
    return genders


def apply_prolog_rules(persons):
    prolog = Prolog()
    prolog.consult("family.pl")  # Load your Prolog file containing facts and rules

    results = []
    for person in persons:
        gender = predict_gender([person])[0][1]
        prolog.assertz(f"gender('{person}', '{gender}')")  # Assert the gender as a fact in Prolog

        # Query Prolog rules using the asserted gender fact
        list(prolog.query(f"father(X, '{person}')"))
        list(prolog.query(f"mother(X, '{person}')"))
        # Add more queries based on your rules

        # Collect the results
        for result in prolog.query(f"father(X, '{person}')"):
            results.append((result["X"], "father"))
        for result in prolog.query(f"mother(X, '{person}')"):
            results.append((result["X"], "mother"))
        # Add more result collection based on your rules

    return results


text = "who is father pf jess"
persons = NER(text)
results = apply_prolog_rules(persons)

for result in results:
    print(f"{result[1].capitalize()} of '{result[0]}' is related to '{result[0]}'")

# import spacy
#
# def NER(text):
#    nlp = spacy.load("en_core_web_sm")
#    doc = nlp(text)
#    entities = []
#    for ent in doc.ents:
#         print(ent)
#         entities.append((ent.text, ent.label_))
#    return entities
# text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak. It is headquartered in Cupertino, California."
# entities = NER(text)
# for entity in entities:
#     print(entity)






