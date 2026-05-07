import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import json
import boto3
from botocore.client import Config

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['file_name']
        img_path = self.root_dir / file_name
        if not img_path.exists():
            img_path = self.root_dir / file_name.replace("train_data/", "test_data/", 1)
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)
bucket = 'models'
key = 'model_base.pth'
s3.download_file(bucket, key, 'model_base.pth')
print('Base model downloaded from S3.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model_base.pth', map_location=device))
model = model.to(device)

train_ds = ImageDataset('ai-vs-human-generated-dataset-hw/Train_2/train.csv',
                        'ai-vs-human-generated-dataset-hw/Train_2', train_transform)
test_ds = ImageDataset('ai-vs-human-generated-dataset-hw/Test_2/test.csv',
                       'ai-vs-human-generated-dataset-hw/Test_2', test_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Finetune Epoch {epoch+1}/5 completed.')

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='weighted')

torch.save(model.state_dict(), 'model_fine.pth')
with open('finetune_metrics.json', 'w') as f:
    json.dump({'test_accuracy': test_acc, 'test_f1': test_f1}, f)

s3.upload_file('model_fine.pth', bucket, 'model_fine.pth')
print(f'Finetuned model saved. Acc: {test_acc:.4f}, F1: {test_f1:.4f}')