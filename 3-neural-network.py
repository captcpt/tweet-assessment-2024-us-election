import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# PyTorch setup models
class TextDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = np.array(y) if y is not None else None  # Convert to numpy array for proper indexing
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx].toarray()[0])
        if self.y is not None:
            return x, torch.LongTensor([self.y[idx]])[0]
        return x

class TextClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.model(x)

# Loading/preparing data
training_df = pd.read_csv("./training_data/training.csv")
unlabeled_df = pd.read_csv("./processed_data/combined_election_data.csv")
training_df['text'] = training_df['text'].fillna('')
unlabeled_df['text'] = unlabeled_df['text'].fillna('')

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode',
    norm='l2'
)

X_train, X_test, y_train, y_test = train_test_split(
    training_df['text'], 
    training_df['poli_label'], 
    test_size=0.2,
    random_state=42,
    stratify=training_df['poli_label']
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

def train_pytorch_model(model, train_loader, criterion, optimizer, device, epochs=15):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# PyTorch setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = TextDataset(X_train_tfidf, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
model = TextClassifier(X_train_tfidf.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training PyTorch model...")
train_pytorch_model(model, train_loader, criterion, optimizer, device)

# Batch Predictions (for memory efficiency)
def predict_in_batches(model, X, batch_size=32):
    predictions = []
    dataset = TextDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    return np.array(predictions)

# Evaluate the model
pytorch_pred = predict_in_batches(model, X_test_tfidf)

print("\nPyTorch Model Results:")
print("Accuracy:", accuracy_score(y_test, pytorch_pred))
print("\nClassification Report:")
print(classification_report(y_test, pytorch_pred))

# Visualize confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, pytorch_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('PyTorch Model Confusion Matrix')
plt.tight_layout()
plt.show()

# Process unlabeled data in batches
unlabeled_tfidf = vectorizer.transform(unlabeled_df['text'])
unlabeled_df['prediction'] = predict_in_batches(model, unlabeled_tfidf)

# Saving results
unlabeled_df.to_csv('./final_data/labeled_election_data.csv', index=False)
print("\nProcessing complete! Results saved to 'labeled_election_data.csv'")