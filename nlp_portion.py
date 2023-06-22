
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_weights('gender_model_weights.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def predict_gender(name):
    encoding = tokenizer([name], truncation=True, padding=True)
    input_dataset = tf.data.Dataset.from_tensor_slices(dict(encoding)).batch(1)
    predictions = model.predict(input_dataset)
    predicted_label = tf.argmax(predictions.logits, axis=1)[0].numpy()
    gender = "male" if predicted_label == 0 else "female"
    return gender


def get_gender_prediction(text):
    gender = predict_gender(text)
    if gender == "male":
        return f"Yes, he is male."
    else:
        return f"Yes, she is female."
text1=input("enter your choice:")
print(get_gender_prediction(text1))
# import tensorflow as tftex
# from transformers import BertTokenizer, TFBertForSequenceClassification
#
#
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model.load_weights('gender_model_weights.h5')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#
# def predict_gender(name):
#     encoding = tokenizer([name], truncation=True, padding=True)
#     input_dataset = tf.data.Dataset.from_tensor_slices(dict(encoding)).batch(1)
#     predictions = model.predict(input_dataset)
#     predicted_label = tf.argmax(predictions.logits, axis=1)[0].numpy()
#     gender = "male" if predicted_label == 0 else "female"
#     return gender
#
#
# text = input("Enter your choice: ")
# answer = predict_gender(text)
#
# print(f"The predicted gender for the person '{text}' is: {answer}")
#
# # import tensorflow as tf
# # from transformers import BertTokenizer, TFBertForSequenceClassification
# # import spacy
# #
# #
# # def NER(text):
# #     nlp = spacy.load("en_core_web_sm")
# #     doc = nlp(text)
# #     persons = []
# #     for ent in doc.ents:
# #         if ent.label_ == "PERSON":
# #             persons.append(ent.text)
# #     return persons
# #
# #
# # model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# # model.load_weights('gender_model_weights.h5')
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# #
# #
# # def predict_gender(names):
# #     genders = []
# #     for name in names:
# #         encoding = tokenizer([name], truncation=True, padding=True)
# #         input_dataset = tf.data.Dataset.from_tensor_slices(dict(encoding)).batch(1)
# #         predictions = model.predict(input_dataset)
# #         predicted_label = tf.argmax(predictions.logits, axis=1)[0].numpy()
# #         gender = "male" if predicted_label == 0 else "female"
# #         genders.append((name, gender))
# #     return genders
# #
# #
# # text = input("enter your choice:")
# # persons = NER(text)
# #
# # genders = predict_gender(persons)
# #
# # for person, gender in genders:
# #     print(f"The predicted gender for the person '{person}' is: {gender}")
