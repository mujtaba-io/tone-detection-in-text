
# what is the tone of text?
# using bert model in pytorch

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset

# load dataset
dataset = load_dataset('csv', data_files={'train': 'train_dataset.csv', 'test': 'test_dataset.csv'})

# dataset has columns 'text' and 'label'
train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
test_texts, test_labels = dataset['test']['text'], dataset['test']['label']

# convert labels to integers
label_map = {label: idx for idx, label in enumerate(set(train_labels))}
train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]

# split the training data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1)

# tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(labels)
    return encodings, labels

train_encodings, train_labels = tokenize_data(train_texts, train_labels)
val_encodings, val_labels = tokenize_data(val_texts, val_labels)
test_encodings, test_labels = tokenize_data(test_texts, test_labels)

# create dataLoader
class ToneDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToneDataset(train_encodings, train_labels)
val_dataset = ToneDataset(val_encodings, val_labels)
test_dataset = ToneDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# model and training Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# training Loop
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels, preds), f1_score(labels, preds, average='weighted')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, val_f1 = eval_epoch(model, val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss}')
    print(f'Val Loss: {val_loss}')
    print(f'Val Accuracy: {val_acc}')
    print(f'Val F1 Score: {val_f1}')

# evaluation
test_loss, test_acc, test_f1 = eval_epoch(model, test_loader)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')
print(f'Test F1 Score: {test_f1}')

# inference
def predict_tone(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).cpu().item()
    
    # convert predicted class back to label
    predicted_label = list(label_map.keys())[predicted_class]
    
    return predicted_label

# test
sample_text = "I am very unhappy with the service."
print(f'Tone of the text: {predict_tone(sample_text)}')

