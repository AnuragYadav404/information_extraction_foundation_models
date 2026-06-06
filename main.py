# Library Imports:
# Import libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW #Improved version of the Adam optimizer
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm #Progress bar

# Check GPU
device = "cpu"

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


# Load the dataset (use the first 100 rows)
df = pd.read_csv("title_conference.csv")

# Encode sentiment labels to numerical values
label_encoder = LabelEncoder()
df['Conference'] = label_encoder.fit_transform(df['Conference'])


# Define custom dataset
class ConferenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts # this takes in a list of conference titles [t1,t2,t3]
        self.labels = labels # corresponding labels of the confernece titles [0,1,3,0 ...]
        self.tokenizer = tokenizer # tokenizer from bert-base
        self.max_len = max_len # max_len? max length of input token we process

    def __len__(self):
        return len(self.texts) # returns number of samples

    def __getitem__(self, idx): # get a training example by index
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    




# Tokenize and split dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = ConferenceDataset(df['Title'].tolist(), df['Conference'].tolist(), tokenizer)

# Split: 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)



# Training loop
epochs = 10
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(train_loader):
        # move input to device 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward() #Compute gradient 
        optimizer.step() #Update weights

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(correct / total)
    print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {train_accuracies[-1]:.4f}")


# Save trained model
torch.save(model.state_dict(), "bert_conference_model.pt")


# # Plot loss and accuracy curves
# plt.plot(train_losses, label='Loss')
# plt.plot(train_accuracies, label='Accuracy')
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.legend()
# plt.show()


with open("train_losses.txt", "w") as f:
    for value in train_losses:
        f.write(f"{value}\n")

with open("train_accuracies.txt", "w") as f:
    for value in train_accuracies:
        f.write(f"{value}\n")