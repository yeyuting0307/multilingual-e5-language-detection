#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('Mike0307/multilingual-e5-language-detection')
model = AutoModelForSequenceClassification.from_pretrained('Mike0307/multilingual-e5-language-detection', num_labels=45)

# %%
import torch

languages = [
    "Arabic", "Basque", "Breton", "Catalan", "Chinese_China", "Chinese_Hongkong", 
    "Chinese_Taiwan", "Chuvash", "Czech", "Dhivehi", "Dutch", "English", 
    "Esperanto", "Estonian", "French", "Frisian", "Georgian", "German", "Greek", 
    "Hakha_Chin", "Indonesian", "Interlingua", "Italian", "Japanese", "Kabyle", 
    "Kinyarwanda", "Kyrgyz", "Latvian", "Maltese", "Mongolian", "Persian", "Polish", 
    "Portuguese", "Romanian", "Romansh_Sursilvan", "Russian", "Sakha", "Slovenian", 
    "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Ukranian", "Welsh"
]

def predict(text, model, tokenizer, device = torch.device('cpu')):
    model.to(device)
    model.eval()
    tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities

def get_topk(probabilities, languages, k=3):
    topk_prob, topk_indices = torch.topk(probabilities, k)
    topk_prob = topk_prob.cpu().numpy()[0].tolist()
    topk_indices = topk_indices.cpu().numpy()[0].tolist()
    topk_labels = [languages[index] for index in topk_indices]
    return topk_prob, topk_labels

text = "你的測試句子"
probabilities = predict(text, model, tokenizer)
topk_prob, topk_labels = get_topk(probabilities, languages)
print(topk_prob, topk_labels)

# [0.999620258808, 0.00025940246996469, 2.7690215574693e-05]
# ['Chinese_Taiwan', 'Chinese_Hongkong', 'Chinese_China']

# %%
