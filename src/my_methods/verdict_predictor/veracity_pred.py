import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments
from torch import nn
from torch.optim import Adam
from nlpaug.augmenter.word import SynonymAug


class ContrastiveStanceDetectionDataset(Dataset):
    def __init__(self, claims, evidences, labels, tokenizer, max_length=512):
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = self.claims[idx]
        evidence = self.evidences[idx]
        label = self.labels[idx]

        claim_encoding = self.tokenizer.encode_plus(
            claim,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        evidence_encoding = self.tokenizer.encode_plus(
            evidence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'claim_input_ids': claim_encoding['input_ids'].flatten(),
            'claim_attention_mask': claim_encoding['attention_mask'].flatten(),
            'evidence_input_ids': evidence_encoding['input_ids'].flatten(),
            'evidence_attention_mask': evidence_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


claims = ["Claim 1", "Claim 2", "Claim 3"]
evidences = ["Evidence 1", "Evidence 2", "Evidence 3"]
labels = [1, 0, 1]  # 1: similar (positive pair), 0: dissimilar (negative pair)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
dataset = ContrastiveStanceDetectionDataset(claims, evidences, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768, 128)

    def forward_one(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]
        return self.fc(pooled_output)

    def forward(self, claim_input_ids, claim_attention_mask, evidence_input_ids, evidence_attention_mask):
        claim_output = self.forward_one(claim_input_ids, claim_attention_mask)
        evidence_output = self.forward_one(evidence_input_ids, evidence_attention_mask)
        return claim_output, evidence_output


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


model = SiameseNetwork()
contrastive_loss = ContrastiveLoss()
optimizer = Adam(model.parameters(), lr=2e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        claim_input_ids = batch['claim_input_ids']
        claim_attention_mask = batch['claim_attention_mask']
        evidence_input_ids = batch['evidence_input_ids']
        evidence_attention_mask = batch['evidence_attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        claim_output, evidence_output = model(claim_input_ids, claim_attention_mask, evidence_input_ids,
                                              evidence_attention_mask)
        loss = contrastive_loss(claim_output, evidence_output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Fine-tuning model
stance_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# Assuming we have train_dataset and eval_dataset ready for fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=stance_model,
    args=training_args,
    train_dataset=dataset,  # replace with your actual dataset
    eval_dataset=dataset  # replace with your actual dataset
)

trainer.train()
trainer.evaluate()
