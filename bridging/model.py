import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from datasets import load_dataset
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal time embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Conv1dBlock(nn.Module):
    """
    A Residual Convolutional Block.
    Input: (Batch, Channels, Length)
    """
    def __init__(self, channels, kernel_size, dropout=0.1):
        super().__init__()
        # Padding logic ensures output length equals input length
        padding = kernel_size // 2 
        
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.Dropout(dropout)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))

class DenoisingModelConv1d(nn.Module):
    """
    Denoising model using 1D Convolutions.
    This treats the latent space like a multi-channel signal.
    """
    def __init__(self, channels, latent_width, kernel_size=3, time_emb_dim=256, num_layers=4):
        super().__init__()
        self.channels = channels
        self.latent_width = latent_width
        
        # Learnable Positional Embedding
        # Shape: (1, Channels, Latent_Width) so it broadcasts over batch dimension
        self.pos_emb = nn.Parameter(torch.randn(1, channels, latent_width) * 0.02)
        
        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Main Backbone
        self.input_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.layers = nn.ModuleList([
            Conv1dBlock(channels, kernel_size=kernel_size) for _ in range(num_layers)
        ])
        
        self.output_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Projection to mix time embedding into channels
        self.time_proj = nn.Linear(time_emb_dim, channels)

    def forward(self, x, t):
        # x: (Batch, Channels, Latent_Width)
        # t: (Batch)
        
        # Add Positional Embedding (Broadcasting)
        x = x + self.pos_emb
        
        # 1. Process Time
        t_emb = self.time_mlp(t)                # (Batch, Time_Dim)
        t_emb = self.time_proj(t_emb)           # (Batch, Channels)
        t_emb = t_emb.unsqueeze(-1)             # (Batch, Channels, 1) - Broadcastable
        
        # 2. Initial Conv
        x = self.input_conv(x)
        
        # 3. Add Time Embedding to features (broadcasting across width)
        x = x + t_emb
        
        # 4. Residual Conv Layers
        for layer in self.layers:
            x = layer(x)
            
        output = self.output_conv(x)
        return output

class DenoisingModelTransformer(nn.Module):
    """
    Denoising model using a Transformer Backbone.
    Treats the latent space as a sequence of length `latent_width`.
    """
    def __init__(self, channels, latent_width, time_emb_dim=256, 
                 d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.latent_width = latent_width
        self.d_model = d_model
        
        # 1. Positional Embedding (Learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, latent_width, d_model) * 0.02)
        
        # 2. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, d_model),
        )

        # 3. Input Projection (Channels -> d_model)
        self.input_proj = nn.Linear(channels, d_model)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Projection (d_model -> Channels)
        self.output_proj = nn.Linear(d_model, channels)

    def forward(self, x, t):
        # x: (Batch, Channels, Latent_Width)
        # t: (Batch)
        
        batch_size = x.shape[0]
        
        # Permute for Transformer: (Batch, Length, Channels)
        x = x.permute(0, 2, 1) 
        
        # Project to d_model: (Batch, Length, d_model)
        x = self.input_proj(x)
        
        # Add Positional Embedding
        x = x + self.pos_emb
        
        # Add Time Embedding (Broadcast over Length)
        t_emb = self.time_mlp(t)                # (Batch, d_model)
        t_emb = t_emb.unsqueeze(1)              # (Batch, 1, d_model)
        
        x = x + t_emb
        
        # Transformer Layers
        x = self.transformer(x)
        
        # Output Projection: (Batch, Length, Channels)
        x = self.output_proj(x)
        
        # Permute back to (Batch, Channels, Length)
        x = x.permute(0, 2, 1)
        
        return x

class DecoderTransformer(nn.Module):
    """
    Transformer-based Decoder to map predicted last hidden states to token logits.
    """
    def __init__(
        self,
        hidden_size,
        max_seq_len,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        # [수정 1] max_seq_len 인자 추가 (위치 임베딩 생성을 위해 필요)
        self.max_seq_len = max_seq_len

        # [수정 2] Positional Embedding 추가
        # Transformer는 위치 정보를 모르므로, 이를 더해주어야 순서 정보를 학습할 수 있습니다.
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        # 초기화 (선택 사항이나 학습 안정성을 위해 권장)
        nn.init.normal_(self.pos_emb, mean=0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hidden_states):
        # hidden_states: (Batch, Seq_Len, Hidden_Size)

        # [수정 3] 위치 정보 더하기
        # 입력 시퀀스 길이에 맞춰 슬라이싱 (보통 max_seq_len과 같겠지만 안전장치)
        seq_len = hidden_states.size(1)
        hidden_states = hidden_states + self.pos_emb[:, :seq_len, :]

        # print(f"DecoderTransformer input shape: {hidden_states.shape}", flush=True) # 디버깅용
        x = self.transformer(hidden_states)
        # print(f"DecoderTransformer output shape: {x.shape}", flush=True) # 디버깅용
        return x


class DiffusionLM(nn.Module):
    def __init__(self, 
                 bert_model_name='bert-base-uncased', 
                 max_seq_len=128,
                 latent_channels=8,
                 latent_width=64,
                 timesteps=1000,
                 model_type='conv', # 'conv' or 'transformer'
                 transformer_config=None, # dict of config for transformer
                 num_diffu_layers=8,
                 kernel_size=3,
                 reg_weight=0.0,
                 time_bias=1.0): # Bias parameter for timestep sampling
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.latent_channels = latent_channels
        self.latent_width = latent_width
        self.timesteps = timesteps
        self.reg_weight = reg_weight
        self.time_bias = time_bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # 1. BERT Setup
        print(f"Loading {bert_model_name}...")
        mlm_model = BertForMaskedLM.from_pretrained(bert_model_name)
        
        self.bert = mlm_model.bert
        self.cls_head = mlm_model.cls
        
        config = self.bert.config
        self.hidden_size = config.hidden_size
        
        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.cls_head.parameters():
            param.requires_grad = True

        # 2. Encoder / Decoder (Reshaping)
        input_flat_dim = self.max_seq_len * self.hidden_size
        latent_flat_dim = latent_channels * latent_width
        
        self.encoder_proj = nn.Sequential(
            nn.Linear(input_flat_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, latent_flat_dim),
            nn.LayerNorm(latent_flat_dim) # Added Norm to stabilize latent space
        )
        
        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_flat_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, input_flat_dim)
        )

        self.decoder_transformer = DecoderTransformer(
            hidden_size=self.hidden_size,
            max_seq_len=self.max_seq_len,
        )

        # 3. Denoising Model
        if model_type == 'conv':
            self.denoise_model = DenoisingModelConv1d(
                channels=latent_channels, 
                latent_width=latent_width,
                num_layers=num_diffu_layers,
                kernel_size=kernel_size
            )
        elif model_type == 'transformer':
            if transformer_config is None:
                # Default config if none provided
                transformer_config = {
                    'd_model': 512,
                    'nhead': 8,
                    'num_layers': 6,
                    'dim_feedforward': 2048,
                    'dropout': 0.1
                }
            
            print(f"Initializing Transformer Denoising Model with config: {transformer_config}")
            self.denoise_model = DenoisingModelTransformer(
                channels=latent_channels,
                latent_width=latent_width,
                **transformer_config
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 4. Diffusion Parameters (UPDATED to Cosine Schedule)
        beta = self.get_cosine_schedule(timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def get_cosine_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        Better for high-noise steps than linear.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def get_latents(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
        
        batch_size = last_hidden_state.shape[0]
        flat_hidden = last_hidden_state.view(batch_size, -1)
        flat_latents = self.encoder_proj(flat_hidden) 
        # print(f"flat_latents.shape: {flat_latents.shape}", flush=True)
        latents = flat_latents.view(batch_size, self.latent_channels, self.latent_width)
        return latents

    def decode_latents(self, latents):
        batch_size = latents.shape[0]
        flat_latents = latents.view(batch_size, -1)
        reconstructed_flat = self.decoder_proj(flat_latents)
        reconstructed_hidden = reconstructed_flat.view(
            batch_size, self.max_seq_len, self.hidden_size
        )
        reconstructed_hidden = self.decoder_transformer(reconstructed_hidden)
        logits = self.cls_head(reconstructed_hidden)
        return logits

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise
    
    def q_sample_no_stochastic(self, x_0, t):
        """
        Deterministic version of q_sample (no noise added).
        Useful for certain evaluation scenarios.
        """
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
        return sqrt_alpha_bar_t * x_0


    def predict_x0_from_noise(self, x_t, t, predicted_noise):
        """
        x_t와 예측된 노이즈로 x_0를 추정합니다.
        DecoderTransformer가 폭주하지 않도록 값을 자릅니다(Clamping).
        """
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]
        
        # [수정 1] 분모 0 방지용 epsilon 상향 (1e-7 -> 1e-5)
        epsilon = 1e-5
        
        # 기본 수식
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / (sqrt_alpha_bar_t + epsilon)
        
        # [수정 2] ★핵심★ Clamping 적용
        # Latent는 정규분포를 따르도록 유도되므로 -5 ~ 5 범위를 벗어나는 것은 비정상적인 값입니다.
        # 이 과정이 없으면 DecoderTransformer 내부 Attention 연산에서 NaN이 발생합니다.
        x_0_pred = torch.clamp(x_0_pred, min=-5.0, max=5.0)
        
        return x_0_pred

    def forward(self, input_ids, attention_mask):
        if input_ids.size(1) != self.max_seq_len:
            raise ValueError(f"Input sequence length must be {self.max_seq_len}")

        # 1. Get Latents
        x_0 = self.get_latents(input_ids, attention_mask)
        batch_size = x_0.shape[0]

        # 2. Timesteps (Biased Sampling)
        # Uniform rand in [0, 1]
        r = torch.rand((batch_size,), device=x_0.device)
        # Apply power: if bias < 1, favors larger values (near 1.0)
        r_biased = r ** self.time_bias 
        t = (r_biased * self.timesteps).long().clamp(0, self.timesteps - 1)

        # t = torch.full((batch_size,), self.timesteps-1, device=x_0.device, dtype=torch.long)
        # print(f"t : {t}", flush=True)

        # 3. Add Noise
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # 4. Predict Noise
        predicted_noise = self.denoise_model(x_t, t)

        # 5. Loss A: MSE Loss (Latent Space)
        mse_loss = F.mse_loss(predicted_noise, noise)
        
        # 6. Loss B: Cross-Entropy Loss (Token Space) - End-to-End
        x_0_pred = self.predict_x0_from_noise(x_t, t, predicted_noise)
        logits = self.decode_latents(x_0_pred)
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = input_ids.view(-1)
        flat_mask = attention_mask.view(-1)
        # print(f"flat_logits.shape: {flat_logits.shape}, flat_labels.shape: {flat_labels.shape}, flat_mask.shape: {flat_mask.shape}", flush=True)
        # print(flat_logits, flush=True)
        # print(flat_labels, flush=True)
        ce_loss_raw = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        ce_loss = (ce_loss_raw * flat_mask).sum() / (flat_mask.sum() + 1e-7)

        # 7. Loss C: Latent Regularization (NEW)
        # Forces latent space to be close to Normal(0, 1) so sampling from noise works
        reg_loss = x_0.pow(2).mean()

        # 8. Total Loss
        total_loss = mse_loss + ce_loss + (self.reg_weight * reg_loss)
        
        # Return breakdown
        return total_loss, {
            "mse": mse_loss.item(),
            "ce": ce_loss.item(),
            "reg": reg_loss.item(),
            "latent_mean": x_0.mean().item(),
            "latent_std": x_0.std().item()
        }

    @torch.no_grad()
    def reconstruct_text(self, input_ids, attention_mask):
        self.eval()
        latents = self.get_latents(input_ids, attention_mask)
        logits = self.decode_latents(latents)
        predicted_ids = torch.argmax(logits, dim=-1)
        return predicted_ids

    @torch.no_grad()
    def sample(self, batch_size, x=None):
        self.eval()
        device = self.beta.device
        
        if x is None:
            x = torch.randn((batch_size, self.latent_channels, self.latent_width), device=device)
        else:
            x = x.to(device)
        
        print(f"Sampling started for {batch_size} sequences...", flush=True)

        for i in reversed(range(self.timesteps)):
            # print(f" Denoising step {i+1}/{self.timesteps}", flush=True)
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            predicted_noise = self.denoise_model(x, t)
            
            alpha_t = self.alpha[t][:, None, None]
            alpha_bar_t = self.alpha_bar[t][:, None, None]
            beta_t = self.beta[t][:, None, None]
            
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean

        logits = self.decode_latents(x)
        predicted_ids = torch.argmax(logits, dim=-1)
        return predicted_ids
        
    @torch.no_grad()
    def check_prediction(self, input_ids, attention_mask, num_samples=5):
        self.eval()
        input_ids = input_ids[:num_samples]
        attention_mask = attention_mask[:num_samples]
        batch_size = input_ids.shape[0]

        x_0 = self.get_latents(input_ids, attention_mask)
        t = torch.linspace(0, self.timesteps - 1, batch_size, device=self.device).long()
        
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        predicted_noise = self.denoise_model(x_t, t)
        
        x_0_pred = self.predict_x0_from_noise(x_t, t, predicted_noise)
        
        logits = self.decode_latents(x_0_pred)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        return predicted_ids, t

def decode_token_ids(token_ids, tokenizer):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if tokenizer.sep_token_id in token_ids:
        sep_idx = token_ids.index(tokenizer.sep_token_id)
        token_ids = token_ids[:sep_idx]
    return tokenizer.decode(token_ids, skip_special_tokens=True)
