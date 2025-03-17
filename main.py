import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from contextlib import nullcontext
from rouge import Rouge
import warnings
warnings.filterwarnings("ignore")

class Config:
    """Cấu hình cho mô hình tóm tắt văn bản tiếng Việt"""
    def __init__(self):
        self.MODEL_NAME = "vinai/phobert-base"
        self.MAX_TEXT_LENGTH = 512
        self.MAX_SUMMARY_LENGTH = 150
        self.BATCH_SIZE = 16
        self.VAL_BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-5
        self.EPOCHS = 1000
        self.SEED = 42
        self.TEACHER_FORCING_RATIO = 0.7
        self.CHECKPOINT_DIR = "checkpoints"
        self.MODEL_PATH = os.path.join(self.CHECKPOINT_DIR, "best_model.pt")
        self.RESULTS_DIR = "results"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cấu hình cho model transformer
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3
        self.NHEAD = 8
        self.DIM_FEEDFORWARD = 2048
        self.DROPOUT = 0.1
        self.WEIGHT_DECAY = 0.01
        self.WARMUP_STEPS = 1000
        self.GRADIENT_ACCUMULATION_STEPS = 4
        self.CLIP_GRAD_NORM = 1.0
        self.USE_FP16 = True
        
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        torch.manual_seed(self.SEED)
        np.random.seed(self.SEED)

class VietnameseTextSummarizationDataset(Dataset):
    """Dataset cho bài toán tóm tắt văn bản tiếng Việt"""
    def __init__(self, texts, summaries, tokenizer, max_text_len=512, max_summary_len=150):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.max_sum_len = min(self.max_summary_len, self.tokenizer.model_max_length)
        self.max_length = min(self.max_text_len, self.tokenizer.model_max_length)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        text_encodings = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        summary_encodings = self.tokenizer(
            summary,
            max_length=self.max_summary_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        if "token_type_ids" in text_encodings:
            token_type_ids = text_encodings["token_type_ids"].squeeze()
        else:
            token_type_ids = torch.zeros_like(text_encodings["input_ids"].squeeze())
        
        return {
            "text_input_ids": text_encodings["input_ids"].squeeze(),
            "text_attention_mask": text_encodings["attention_mask"].squeeze(),
            "text_token_type_ids": token_type_ids,
            "summary_input_ids": summary_encodings["input_ids"].squeeze(),
            "summary_attention_mask": summary_encodings["attention_mask"].squeeze()
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Mã hóa vị trí cho Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    """Lớp Encoder tùy chỉnh cho Transformer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoderLayer(nn.Module):
    """Lớp Decoder tùy chỉnh cho Transformer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, attention_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, attention_weights

class EnhancedTransformerModel(nn.Module):
    """Mô hình Transformer nâng cao cho tóm tắt văn bản tiếng Việt"""
    def __init__(self, tokenizer, phobert_model, num_encoder_layers=4, num_decoder_layers=4, 
                 nhead=8, dim_feedforward=2048, dropout=0.1):
        super(EnhancedTransformerModel, self).__init__()
        
        self.tokenizer = tokenizer
        self.encoder_base = phobert_model
        self.d_model = self.encoder_base.config.hidden_size
        self.vocab_size = tokenizer.vocab_size
        
        # Encoding layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoding layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output generation
        self.output_layer = nn.Linear(self.d_model, self.vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Paper: "Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence" (Chi Sun et al., 2019)
        # Pre-training the embedding layer with the tokenizer's vocabulary
        if hasattr(self.encoder_base, 'embeddings') and hasattr(self.encoder_base.embeddings, 'word_embeddings'):
            self.embedding.weight.data.copy_(self.encoder_base.embeddings.word_embeddings.weight.data[:64000])
        
    def encode(self, text_input_ids, text_attention_mask):
        batch_size = text_input_ids.size(0)
        seq_length = text_input_ids.size(1)
        
        text_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long, device=text_input_ids.device)

        # Get PHOBERT base output
        encoder_outputs = self.encoder_base(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            return_dict=True
        )
        
        # Process through additional transformer encoder layers
        encoder_hidden = encoder_outputs.last_hidden_state
        
        for layer in self.encoder_layers:
            encoder_hidden = layer(encoder_hidden, src_key_padding_mask=~text_attention_mask.bool())
        
        return encoder_hidden
    
    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        # Embedding + positional encoding
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        attention_weights_list = []
        
        # Process through transformer decoder layers
        for layer in self.decoder_layers:
            tgt, attention_weights = layer(
                tgt, memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            attention_weights_list.append(attention_weights)
        
        return tgt, attention_weights_list
    
    def forward(self, text_input_ids, text_attention_mask, summary_input_ids=None, summary_attention_mask=None, teacher_forcing_ratio=1.0):
        # Encoding
        memory = self.encode(text_input_ids, text_attention_mask)
        
        if summary_input_ids is not None:
            # Training mode
            target_length = summary_input_ids.size(1)
            batch_size = text_input_ids.size(0)
            
            # Prepare target sequences for decoder (shift right)
            tgt = summary_input_ids[:, :-1]  # Remove last token
            tgt_padding_mask = ~summary_attention_mask[:, :-1].bool()
            memory_key_padding_mask = ~text_attention_mask.bool()
            
            # Create causal mask to prevent attending to future positions
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(text_input_ids.device)
            
            # Decoder
            decoder_output, _ = self.decode(
                tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask
            )
            
            # Output projection
            outputs = self.output_layer(decoder_output)
            
            # Calculate loss
            shift_outputs = outputs.contiguous()
            shift_targets = summary_input_ids[:, 1:].contiguous()  # Remove first token (usually CLS)
            
            loss = self.criterion(
                shift_outputs.view(-1, shift_outputs.size(-1)),
                shift_targets.view(-1)
            )
            
            return outputs, loss
        
        else:
            # Inference mode
            batch_size = text_input_ids.size(0)
            max_length = 150
            memory_key_padding_mask = ~text_attention_mask.bool()
            
            # Start with CLS token
            decoded_ids = torch.full((batch_size, 1), self.tokenizer.cls_token_id, 
                                    dtype=torch.long, device=text_input_ids.device)
            
            for i in range(max_length - 1):
                tgt = decoded_ids
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(text_input_ids.device)
                tgt_padding_mask = None  # All tokens are valid during generation
                
                # Decode one step
                decoder_output, _ = self.decode(
                    tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask
                )
                
                # Predict next token
                next_token_logits = self.output_layer(decoder_output[:, -1])
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to output
                decoded_ids = torch.cat([decoded_ids, next_token], dim=1)
                
                # Stop if all sequences have generated an EOS token
                if (next_token == self.tokenizer.sep_token_id).all():
                    break
            
            return decoded_ids, None
    
    def generate_square_subsequent_mask(self, size):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class VietnameseSummarizer:
    """Pipeline tóm tắt văn bản tiếng Việt"""
    def __init__(self, config=None):
        if config is None:
            self.config = Config()
        else:
            self.config = config
            
        self.tokenizer = None
        self.model = None
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        
        print(f"Khởi tạo pipeline tóm tắt văn bản với device: {self.config.DEVICE}")
        
    def load_tokenizer_and_model(self):
        """Tải tokenizer và mô hình"""
        print("Đang tải tokenizer và model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME, use_fast=False)
        phobert = AutoModel.from_pretrained(self.config.MODEL_NAME)

        self.model = EnhancedTransformerModel(
            tokenizer=self.tokenizer, 
            phobert_model=phobert,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.NUM_DECODER_LAYERS,
            nhead=self.config.NHEAD,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            dropout=self.config.DROPOUT
        )
        self.model.tokenizer = self.tokenizer
        self.model.to(self.config.DEVICE)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Tổng số tham số: {total_params:,}")
        print(f"Số tham số có thể huấn luyện: {trainable_params:,}")
        
        return self
    
    def prepare_data(self, df, text_col="original_summary", summary_col="generated_summary"):
        """Chuẩn bị dữ liệu từ DataFrame"""
        df = df.dropna(subset=[text_col, summary_col])
        print(f"Số lượng mẫu sau khi làm sạch: {len(df)}")

        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=self.config.SEED
        )
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

        train_dataset = VietnameseTextSummarizationDataset(
            texts=train_df[text_col].tolist(),
            summaries=train_df[summary_col].tolist(),
            tokenizer=self.tokenizer,
            max_text_len=self.config.MAX_TEXT_LENGTH,
            max_summary_len=self.config.MAX_SUMMARY_LENGTH
        )
        
        val_dataset = VietnameseTextSummarizationDataset(
            texts=val_df[text_col].tolist(),
            summaries=val_df[summary_col].tolist(),
            tokenizer=self.tokenizer,
            max_text_len=self.config.MAX_TEXT_LENGTH,
            max_summary_len=self.config.MAX_SUMMARY_LENGTH
        )

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        return train_dataloader, val_dataloader, train_df, val_df
    
    def train(self, train_dataloader, val_dataloader):
        """Huấn luyện mô hình"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.config.LEARNING_RATE)
        
        num_training_steps = len(train_dataloader) * self.config.EPOCHS
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=num_training_steps
        )

        best_val_loss = float("inf")
        progress_bar = tqdm(range(num_training_steps), desc="Training")
        
        # Enable fp16 training if config specifies
        scaler = torch.cuda.amp.GradScaler() if self.config.USE_FP16 else None
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            for i, batch in enumerate(train_dataloader):
                with torch.cuda.amp.autocast() if self.config.USE_FP16 else nullcontext():
                    text_input_ids = batch["text_input_ids"].to(self.config.DEVICE)
                    text_attention_mask = batch["text_attention_mask"].to(self.config.DEVICE)
                    summary_input_ids = batch["summary_input_ids"].to(self.config.DEVICE)
                    summary_attention_mask = batch["summary_attention_mask"].to(self.config.DEVICE)
                    
                    _, loss = self.model(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        summary_input_ids=summary_input_ids,
                        summary_attention_mask=summary_attention_mask,
                        teacher_forcing_ratio=self.config.TEACHER_FORCING_RATIO
                    )
                    
                    # Gradient accumulation
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass with scaling if fp16 is enabled
                if self.config.USE_FP16:
                    scaler.scale(loss).backward()
                    if (i + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_GRAD_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (i + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_GRAD_NORM)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
                progress_bar.update(1)
                
                # Log metrics periodically
                if i % 100 == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}, Batch {i}, LR: {lr:.2e}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_dataloader)

            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    text_input_ids = batch["text_input_ids"].to(self.config.DEVICE)
                    text_attention_mask = batch["text_attention_mask"].to(self.config.DEVICE)
                    summary_input_ids = batch["summary_input_ids"].to(self.config.DEVICE)
                    summary_attention_mask = batch["summary_attention_mask"].to(self.config.DEVICE)
                    
                    _, loss = self.model(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        summary_input_ids=summary_input_ids,
                        summary_attention_mask=summary_attention_mask
                    )
                    
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_dataloader)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                print(f"Đã lưu model tốt nhất với val_loss = {best_val_loss:.4f}")

        self.plot_training_history()
        
        return self
    
    def save_model(self, path=None):
        """Lưu mô hình"""
        if path is None:
            path = self.config.MODEL_PATH
            
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config
        }, path)
        
        return self
    
    def load_model(self, path=None):
        """Tải mô hình đã huấn luyện"""
        if path is None:
            path = self.config.MODEL_PATH
            
        if not os.path.exists(path):
            print(f"Không tìm thấy model tại {path}. Vui lòng huấn luyện model trước.")
            return self
            
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Đã tải model từ {path}")
        
        return self
    
    def plot_training_history(self):
        """Vẽ biểu đồ lịch sử huấn luyện"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.RESULTS_DIR, "training_history.png"))
        plt.close()
        
    def generate_summary(self, text, max_text_len=None):
        """Tóm tắt văn bản"""
        if max_text_len is None:
            max_text_len = self.config.MAX_TEXT_LENGTH
            
        self.model.eval()

        text_encodings = self.tokenizer(
            text, 
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.config.DEVICE)

        with torch.no_grad():
            generated_ids, _ = self.model(
                text_input_ids=text_encodings["input_ids"],
                text_attention_mask=text_encodings["attention_mask"]
            )

        generated_summaries = []
        for ids in generated_ids:
            filtered_ids = ids[(ids != self.tokenizer.pad_token_id) & 
                             (ids != self.tokenizer.cls_token_id) & 
                             (ids != self.tokenizer.sep_token_id)]
            
            summary = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
            generated_summaries.append(summary)
        
        return generated_summaries[0]  
    
    def evaluate(self, val_df, text_col="text", summary_col="summary", num_examples=5):
        """Đánh giá mô hình trên tập validation"""
        assert num_examples <= len(val_df), f"Số lượng mẫu đánh giá ({num_examples}) không thể lớn hơn kích thước tập validation ({len(val_df)})"

        sample_df = val_df.sample(num_examples, random_state=self.config.SEED)
        
        results = []
        rouge = Rouge()
        
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating"):
            text = row[text_col]
            original_summary = row[summary_col]
            
            generated_summary = self.generate_summary(text)

            if original_summary and generated_summary:
                try:
                    rouge_scores = rouge.get_scores(generated_summary, original_summary)[0]
                except:
                    rouge_scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
            else:
                rouge_scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
                
            results.append({
                "text": text,
                "original_summary": original_summary,
                "generated_summary": generated_summary,
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"]
            })

        avg_rouge1 = np.mean([r["rouge-1"] for r in results])
        avg_rouge2 = np.mean([r["rouge-2"] for r in results])
        avg_rougeL = np.mean([r["rouge-l"] for r in results])
        
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(os.path.join(self.config.RESULTS_DIR, "evaluation_results.csv"), index=False)
        
        return result_df

def run_pipeline():
    """Chạy toàn bộ pipeline tóm tắt văn bản tiếng Việt"""
    print("=== BẮT ĐẦU PIPELINE TÓM TẮT VĂN BẢN TIẾNG VIỆT ===")
    
    config = Config()

    summarizer = VietnameseSummarizer(config)

    summarizer.load_tokenizer_and_model()

    csv_path = "tokenized_data.csv"
    df = pd.read_csv(csv_path)
    print(f"Đã đọc {len(df)} mẫu từ dataset")

    # Xác định các cột phù hợp nhất cho huấn luyện
    text_col = None
    summary_col = None
    
    # Ưu tiên các cột đã được làm sạch
    if "cleaned_content" in df.columns and "cleaned_summary" in df.columns:
        text_col, summary_col = "cleaned_content", "cleaned_summary"
        print("Sử dụng cột cleaned_content và cleaned_summary cho huấn luyện")
    elif "tokenized_content" in df.columns and "tokenized_summary" in df.columns:
        text_col, summary_col = "tokenized_content", "tokenized_summary"
        print("Sử dụng cột tokenized_content và tokenized_summary cho huấn luyện")
    elif "Nội dung" in df.columns and "Tóm tắt" in df.columns:
        text_col, summary_col = "Nội dung", "Tóm tắt"
        print("Sử dụng cột Nội dung và Tóm tắt cho huấn luyện")
    elif "content" in df.columns and "summary" in df.columns:
        text_col, summary_col = "content", "summary"
        print("Sử dụng cột content và summary cho huấn luyện")
    else:
        print("Không tìm thấy cặp cột text/summary chuẩn. Sử dụng Nội dung/Tóm tắt mặc định.")
        text_col, summary_col = "Nội dung", "Tóm tắt"

    # Kiểm tra nulls và duplicate
    null_count_text = df[text_col].isna().sum()
    null_count_summary = df[summary_col].isna().sum()
    duplicate_count = df.duplicated([text_col, summary_col]).sum()
    
    print(f"Số lượng giá trị null trong cột {text_col}: {null_count_text}")
    print(f"Số lượng giá trị null trong cột {summary_col}: {null_count_summary}")
    print(f"Số lượng bản ghi trùng lặp: {duplicate_count}")
    
    # Thống kê độ dài
    text_lengths = df[text_col].str.len()
    summary_lengths = df[summary_col].str.len()
    
    print(f"Độ dài trung bình của {text_col}: {text_lengths.mean():.2f} ký tự")
    print(f"Độ dài trung bình của {summary_col}: {summary_lengths.mean():.2f} ký tự")
    print(f"Tỷ lệ nén trung bình: {summary_lengths.mean() / text_lengths.mean():.2%}")

    # Loại bỏ các mẫu có summary quá dài so với text gốc
    length_ratio = summary_lengths / text_lengths
    suspicious_samples = df[length_ratio > 0.8].shape[0]
    print(f"Số lượng mẫu có tỷ lệ nén > 0.8 (có thể không phải tóm tắt thực sự): {suspicious_samples}")
    
    # Loại bỏ các mẫu có summary quá ngắn
    too_short_summaries = df[summary_lengths < 10].shape[0]
    print(f"Số lượng mẫu có summary quá ngắn (< 10 ký tự): {too_short_summaries}")
    
    # Lọc dữ liệu cho huấn luyện
    filtered_df = df[
        (length_ratio <= 0.8) &                # Tóm tắt không quá dài so với văn bản gốc
        (summary_lengths >= 10) &              # Tóm tắt đủ dài
        (df[text_col].notna()) &               # Không có giá trị null
        (df[summary_col].notna())              # Không có giá trị null
    ]
    
    print(f"Số lượng mẫu sau khi lọc: {len(filtered_df)} (giảm {len(df) - len(filtered_df)} mẫu)")

    # Chuẩn bị dữ liệu
    train_dataloader, val_dataloader, train_df, val_df = summarizer.prepare_data(
        filtered_df, text_col=text_col, summary_col=summary_col
    )

    # Huấn luyện mô hình
    summarizer.train(train_dataloader, val_dataloader)

    # Đánh giá mô hình
    results = summarizer.evaluate(val_df, text_col=text_col, summary_col=summary_col, num_examples=5)

    print("\n=== MẪU KẾT QUẢ ===")
    for i, row in results.iterrows():
        print(f"\n--- Ví dụ {i+1} ---")
        print(f"Văn bản gốc: {row['text'][:200]}...")
        print(f"Tóm tắt gốc: {row['original_summary']}")
        print(f"Tóm tắt tạo ra: {row['generated_summary']}")
        print(f"ROUGE-1: {row['rouge-1']:.4f}, ROUGE-2: {row['rouge-2']:.4f}, ROUGE-L: {row['rouge-l']:.4f}")
    
    print("\n=== HOÀN TẤT PIPELINE TÓM TẮT VĂN BẢN TIẾNG VIỆT ===")
    
    return summarizer

if __name__ == "__main__":
    summarizer = run_pipeline()