# %%
# ==================== Prepare Data ====================
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

dataset = load_dataset("common_language")

languages = [
    "Arabic", "Basque", "Breton", "Catalan", "Chinese_China", "Chinese_Hongkong", 
    "Chinese_Taiwan", "Chuvash", "Czech", "Dhivehi", "Dutch", "English", 
    "Esperanto", "Estonian", "French", "Frisian", "Georgian", "German", "Greek", 
    "Hakha_Chin", "Indonesian", "Interlingua", "Italian", "Japanese", "Kabyle", 
    "Kinyarwanda", "Kyrgyz", "Latvian", "Maltese", "Mongolian", "Persian", "Polish", 
    "Portuguese", "Romanian", "Romansh_Sursilvan", "Russian", "Sakha", "Slovenian", 
    "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Ukranian", "Welsh"
]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')

class LanguageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]
        return input_ids, attention_mask, label

texts = [item['sentence'] for item in dataset['train']]
labels = [item['language'] for item in dataset['train']]
train_dataset = LanguageDataset(texts, labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ==================== Train ====================
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import os

model = AutoModelForSequenceClassification.from_pretrained('intfloat/multilingual-e5-base', num_labels=45)

device = torch.device('mps')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

def train_model(model, data_loader, optimizer, num_epochs, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.2f}")

        # Save model
        model_save_name = f'model_epoch_{epoch+1}.pt'
        save_path_epoch = os.path.join(save_path, model_save_name)
        torch.save(model.state_dict(), save_path_epoch)
        print(f"Model saved to {save_path_epoch}")

train_model(model, train_loader, optimizer, num_epochs=10, save_path='models/')

# %%
# ==================== Test ====================
test_texts = [item['sentence'] for item in dataset['test']]
test_labels = [item['language'] for item in dataset['test']]
test_dataset = LanguageDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate(model, data_loader, tokenizer):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

predictions, labels = evaluate(model, test_loader, tokenizer)

overall_accuracy = accuracy_score(labels, predictions)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

class_report = classification_report(labels, predictions, target_names=languages)
print(class_report)


#%%
# ==================== Inference ====================
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

#%%
# ============ To HuggingFace Model ============
hf_repo = 'Mike0307/multilingual-e5-language-detection'
model.push_to_hub(hf_repo)
tokenizer.push_to_hub(hf_repo)
# %%
