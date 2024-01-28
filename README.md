---
license: apache-2.0
datasets:
- common_language
language:
- ar
- eu
- br
- ca
- zh
- cv
- cs
- nl
- en
- eo
- et
- fr
- ka
- de
- el
- id
- ia
- it
- ja
- rw
- ky
- lv
- mt
- mn
- fa
- pl
- pt
- ro
- rm
- ru
- sl
- es
- sv
- ta
- tt
- tr
- uk
- cy
metrics:
- accuracy
- precision
- recall
- f1
tags:
- language-detection
- Frisian
- Dhivehi
- Hakha_Chin
- Kabyle
- Sakha
---


### Overview
This model supports the detection of **45** languages, and it's fine-tuned using **multilingual-e5-base** model on the **common-language** dataset.<br>
The overall accuracy is **98.37%**, and more evaluation results are shown the below.

### Download the model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('Mike0307/multilingual-e5-language-detection')
model = AutoModelForSequenceClassification.from_pretrained('Mike0307/multilingual-e5-language-detection', num_labels=45)
```

### Example of language detection
```python
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
```

### Evaluation Results
The test datasets refers to the **common_language** test datasets.

|index| language | precision | recall | f1-score | support |
| --- | --- | --- | ---| --- | --- |
|0|Arabic|1.00|1.00|1.00|151|
|1|           Basque   |    0.99   |   1.00   |  1.00   |     111|
|2|           Breton   |    1.00   |   0.90   |   0.95  |     252|
|3|          Catalan   |    0.96   |   0.99   |   0.97  |      96|
|4|    Chinese_China   |    0.98   |   1.00   |   0.99  |     100|
|5| Chinese_Hongkong   |    0.97   |   0.87   |   0.92  |     115|
|6|   Chinese_Taiwan   |    0.92   |   0.98   |   0.95  |     170|
|7|          Chuvash   |    0.98   |   1.00   |   0.99  |     137|
|8|            Czech   |    0.98   |   1.00   |   0.99  |     128|
|9|          Dhivehi   |    1.00   |   1.00   |   1.00  |     111|
|10|            Dutch   |    0.99   |   1.00   |   0.99  |     144|
|11|          English   |    0.96   |   1.00   |   0.98  |      98|
|12|        Esperanto   |    0.98   |   0.98   |   0.98  |     107|
|13|         Estonian   |    1.00   |   0.99   |   0.99  |      93|
|14|           French   |    0.95   |   1.00   |   0.98  |     106|
|15|          Frisian   |    1.00   |   0.98   |   0.99  |     117|
|16|         Georgian   |    1.00   |   1.00   |   1.00  |     110|
|17|           German   |    1.00   |   1.00   |   1.00  |     101|
|18|            Greek   |    1.00   |   1.00   |   1.00  |     153|
|19|       Hakha_Chin   |    0.99   |   1.00   |   0.99  |     202|
|20|       Indonesian   |    0.99   |   0.99   |   0.99  |     150|
|21|      Interlingua   |    0.96   |   0.97   |   0.96  |     182|
|22|          Italian   |    0.99   |   0.94   |   0.96  |     100|
|23|         Japanese   |    1.00   |   1.00   |   1.00  |     144|
|24|           Kabyle   |    1.00   |   0.96   |   0.98  |     156|
|25|      Kinyarwanda   |    0.97   |   1.00   |   0.99  |     103|
|26|           Kyrgyz   |    0.98   |   1.00   |   0.99  |     129|
|27|          Latvian   |    0.98   |   0.98   |   0.98  |     171|
|28|          Maltese   |    0.99   |   0.98   |   0.98  |     152|
|29|        Mongolian   |    1.00   |   1.00   |   1.00  |     112|
|30|          Persian   |    1.00   |   1.00   |   1.00  |     123|
|31|           Polish   |    0.91   |   0.99   |   0.95  |     128|
|32|       Portuguese   |    0.94   |   0.99   |   0.96  |     124|
|33|         Romanian   |    1.00   |   1.00   |   1.00  |     152|
|34|Romansh_Sursilvan   |    0.99   |   0.95   |   0.97  |     106|
|35|          Russian   |    0.99   |   0.99   |   0.99  |     100|
|36|            Sakha   |    0.99   |   1.00   |   1.00  |     105|
|37|        Slovenian   |    0.99   |   1.00   |   1.00  |     166|
|38|          Spanish   |    0.96   |   0.95   |   0.95  |      94|
|39|          Swedish   |    0.99   |   1.00   |   0.99  |     190|
|40|            Tamil   |    1.00   |   1.00   |   1.00  |     135|
|41|            Tatar   |    1.00   |   0.96   |   0.98  |     173|
|42|          Turkish   |    1.00   |   1.00   |   1.00  |     137|
|43|         Ukranian   |    0.99   |   1.00   |   1.00  |     126|
|44|            Welsh   |    0.98   |   1.00   |   0.99  |     103|
||
||        *macro avg*   |    0.98 |     0.99 |     0.98   |   5963|
||     *weighted avg*   |    0.98 |     0.98 |     0.98   |   5963|
|||
||  *overall accuracy*   |         |          |     0.9837   |   5963|


