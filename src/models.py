import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

# --- PyTorch Dataset for the FT-Transformer ---
class TitanicFTDataset(pl.LightningDataModule):
    def __init__(self, X_num, X_cat, y=None):
        super().__init__()
        self.X_num = torch.tensor(X_num.values, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat.values, dtype=torch.int64)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        else:
            return self.X_num[idx], self.X_cat[idx]


# --- FT-Transformer Building Blocks ---

class FeatureTokenizer(nn.Module):
    """
    Tokenizes numerical and categorical features into a sequence of embeddings.
    """
    def __init__(self, num_numerical_features, cat_cardinalities, embed_dim):
        super().__init__()
        self.num_linears = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_numerical_features)])
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_classes, embed_dim) for num_classes in cat_cardinalities])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        
        # Tokenize numerical features
        num_token_list = []
        x_num_unsqueezed = x_num.unsqueeze(-1)
        for i, linear_layer in enumerate(self.num_linears):
            num_token_list.append(linear_layer(x_num_unsqueezed[:, i, :]))
        
        # Tokenize categorical features
        cat_token_list = []
        x_cat_long = x_cat.long()
        for i, embedding_layer in enumerate(self.cat_embeddings):
            cat_token_list.append(embedding_layer(x_cat_long[:, i]))
        
        num_tokens = torch.stack(num_token_list, dim=1)
        cat_tokens = torch.stack(cat_token_list, dim=1)
        
        # Concatenate all feature tokens
        feature_tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        
        # Prepend the [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)
        
        return tokens

class TransformerBlock(nn.Module):
    """
    A standard Transformer block with Multi-Head Attention and a Feed-Forward Network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerEncoder(nn.Module):
    """
    A stack of TransformerBlocks.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PredictionHead(nn.Module):
    """
    The final prediction head that takes the [CLS] token output.
    """
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
    def forward(self, x):
        # Use the output of the CLS token (the first token) for prediction
        cls_token_output = x[:, 0]
        return self.fc(cls_token_output)

# --- Main LightningModule for the FT-Transformer ---
class FTTransformerFromScratch(pl.LightningModule):
    """
    The main PyTorch Lightning module that combines all the building blocks
    and defines the training, validation, and prediction logic.
    """
    def __init__(self, num_numerical, cat_cardinalities, embed_dim, num_heads, ff_dim, num_layers, lr, weight_decay, dropout):
        super().__init__()
        self.save_hyperparameters()
        
        # Build the model from the blocks
        self.tokenizer = FeatureTokenizer(num_numerical, cat_cardinalities, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.head = PredictionHead(embed_dim, 1) # Output dim is 1 for binary classification
        
        # Define loss function and metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        x = self.encoder(x)
        output = self.head(x)
        return output
        
    def training_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        logits = self(x_num, x_cat)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        logits = self(x_num, x_cat)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_num, x_cat = batch
        logits = self(x_num, x_cat)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
