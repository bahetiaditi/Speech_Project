import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from speechbrain.inference.separation import SepformerSeparation
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import math
import librosa
from huggingface_hub import login
from functools import reduce
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from itertools import permutations
from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def setup_huggingface_auth():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not found. Authentication may fail.")
        return False
    login(token=hf_token)
    return True

def get_module_by_name(model, access_string):
    names = access_string.split('.')
    return reduce(getattr, names, model)

class LoRAMultiheadAttention(nn.Module):
    def __init__(self, original_mha, rank=4, alpha=16):
        super().__init__()
        self.original_mha = original_mha
        device = next(original_mha.parameters()).device
        
        for param in self.original_mha.parameters():
            param.requires_grad = False
        
        embed_dim = self.original_mha.embed_dim
        
        self.q_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.q_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        self.k_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.k_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        self.v_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.v_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        self.out_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.out_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.q_lora_B)
        nn.init.kaiming_uniform_(self.k_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.k_lora_B)
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.v_lora_B)
        nn.init.kaiming_uniform_(self.out_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.out_lora_B)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        device = query.device
        
        if self.q_lora_A.device != device:
            self.q_lora_A = self.q_lora_A.to(device)
            self.q_lora_B = self.q_lora_B.to(device)
            self.k_lora_A = self.k_lora_A.to(device)
            self.k_lora_B = self.k_lora_B.to(device)
            self.v_lora_A = self.v_lora_A.to(device)
            self.v_lora_B = self.v_lora_B.to(device)
            self.out_lora_A = self.out_lora_A.to(device)
            self.out_lora_B = self.out_lora_B.to(device)
        
        attn_output, attn_weights = self.original_mha(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            need_weights=need_weights, 
            attn_mask=attn_mask
        )
        
        q_lora = query @ self.q_lora_A.T @ self.q_lora_B.T * self.scaling
        k_lora = key @ self.k_lora_A.T @ self.k_lora_B.T * self.scaling
        v_lora = value @ self.v_lora_A.T @ self.v_lora_B.T * self.scaling
        
        lora_contribution = q_lora + k_lora + v_lora
        attn_output = attn_output + lora_contribution
        
        return attn_output, attn_weights

def apply_lora_to_multihead_attention(model, rank=4, alpha=16):
    replaced_modules = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(f"Found MultiheadAttention module: {name}")
        
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                try:
                    parent_module = get_module_by_name(model, parent_name)
                    lora_attention = LoRAMultiheadAttention(module, rank=rank, alpha=alpha)
                    lora_attention = lora_attention.to(device)
                    setattr(parent_module, child_name, lora_attention)
                    replaced_modules[name] = lora_attention
                    print(f"Applied LoRA to {name}")
                except Exception as e:
                    print(f"Error applying LoRA to {name}: {e}")
    
    print(f"Applied LoRA to {len(replaced_modules)} attention modules")
    return model, replaced_modules

class MixedAudioDataset(Dataset):
    def __init__(self, audio_folder, num_sources=3, segment_length=4.0, sample_rate=16000, normalize_method='unit_max'):
        self.audio_files = [os.path.join(audio_folder, f) 
                           for f in os.listdir(audio_folder) 
                           if f.endswith('.flac')]
        self.num_sources = num_sources
        self.segment_samples = int(segment_length * sample_rate)
        self.sample_rate = sample_rate
        self.normalize_method = normalize_method
        
        if len(self.audio_files) < num_sources:
            raise ValueError(f"Need at least {num_sources} audio files, found {len(self.audio_files)}")
        
        print(f"Found {len(self.audio_files)} audio files. Each mix will use {num_sources} sources.")
        print(f"Using normalization method: {normalize_method}")

    def __len__(self):
        return len(self.audio_files)  

    def _load_audio(self, path):
        
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.segment_samples)
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        elif waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
            
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        return waveform

    def _normalize_audio(self, waveform):
        if self.normalize_method == 'unit_max':
            max_val = waveform.abs().max() + 1e-8
            return 0.9 * (waveform / max_val)
        elif self.normalize_method == 'rms':
            rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
            target_rms = 0.05  
            return waveform * (target_rms / rms)
        else:  # 'none'
            return waveform

    def __getitem__(self, idx):
        source_indices = random.sample(range(len(self.audio_files)), self.num_sources)
        sources = []
        
        for src_idx in source_indices:
            waveform = self._load_audio(self.audio_files[src_idx])
            
            if waveform.shape[0] >= self.segment_samples:
                start = random.randint(0, waveform.shape[0] - self.segment_samples)
                waveform = waveform[start:start + self.segment_samples]
            else:
                padding = self.segment_samples - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            waveform = self._normalize_audio(waveform)
            sources.append(waveform)
        
        sources = torch.stack(sources)
        
        mixture = sources.sum(dim=0)
        
        if mixture.abs().max() > 0.99:
            mixture = 0.99 * mixture / mixture.abs().max()
        
        return mixture, sources

    @staticmethod
    def collate_fn(batch):
        mixtures, sources = zip(*batch)
        
        max_len = max(m.shape[0] for m in mixtures)
        
        def pad(x):
            if x.shape[0] < max_len:
                return torch.nn.functional.pad(x, (0, max_len - x.shape[0]))
            return x
        
        mixtures = torch.stack([pad(m) for m in mixtures])
        sources = torch.stack([torch.stack([pad(s) for s in src]) for src in sources])
        
        return mixtures, sources


def si_sdr_loss(est_sources, target_sources, eps=1e-8, permutation_invariant=True):
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape
    
    si_sdrs = []
    for b in range(batch_size):
        est_b = est_sources[b] 
        target_b = target_sources[b]  
        
        est_b = est_b - est_b.mean(dim=1, keepdim=True)
        target_b = target_b - target_b.mean(dim=1, keepdim=True)
        
        if permutation_invariant and n_sources > 1:
            all_perms = list(permutations(range(n_sources)))
            
            sdr_perms = []
            for perm in all_perms:
                est_perm = est_b[list(perm)]
                
                sdr_sum = 0
                for s in range(n_sources):
                    s_target = (torch.sum(est_perm[s] * target_b[s]) * target_b[s]) / (torch.sum(target_b[s]**2) + eps)
                    
                    e_noise = est_perm[s] - s_target
                    
                    si_sdr_s = 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + eps) + eps)
                    sdr_sum += si_sdr_s
                
                sdr_perms.append(sdr_sum)
            
            si_sdrs.append(torch.max(torch.stack(sdr_perms)))
        else:
            sdr_sum = 0
            for s in range(n_sources):
                s_target = (torch.sum(est_b[s] * target_b[s]) * target_b[s]) / (torch.sum(target_b[s]**2) + eps)
                
                e_noise = est_b[s] - s_target
                
                si_sdr_s = 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + eps) + eps)
                sdr_sum += si_sdr_s
            
            si_sdrs.append(sdr_sum / n_sources)  
    
    si_sdr = torch.stack(si_sdrs).mean()
    
    return si_sdr

def normalize_sources(est_sources, max_amplitude=0.9):
    normalized = est_sources.clone()
    
    max_vals = normalized.abs().max(dim=-1, keepdim=True)[0]

    max_vals = torch.clamp(max_vals, min=1e-8)
    
    normalized = normalized * (max_amplitude / max_vals)
    
    return normalized

def spectral_coherence_index_loss(est_sources, target_sources, n_fft=512, hop_length=128, eps=1e-8, permutation_invariant=True):
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape
    
    window = torch.hann_window(n_fft).to(est_sources.device)
    
    sci_values = []
    for b in range(batch_size):
        est_b = est_sources[b]  # [n_sources, time]
        target_b = target_sources[b]  # [n_sources, time]
        
        if permutation_invariant and n_sources > 1:

            all_perms = list(permutations(range(n_sources)))
            
            sci_perms = []
            for perm in all_perms:
                est_perm = est_b[list(perm)]
                
                # Calculate SCI for this permutation
                sci_sum = 0
                for s in range(n_sources):
                    # Compute STFTs
                    X_ref = torch.stft(target_b[s], n_fft=n_fft, hop_length=hop_length, window=window, 
                                    return_complex=True)
                    X_est = torch.stft(est_perm[s], n_fft=n_fft, hop_length=hop_length, window=window, 
                                    return_complex=True)
                    
                    # Compute SCI frame by frame
                    num_frames = X_ref.shape[1]
                    sci_frames = []
                    
                    for t in range(num_frames):
                        # Get magnitudes for current frame
                        X_ref_frame = X_ref[:, t]
                        X_est_frame = X_est[:, t]
                        
                        # Compute coherence for this frame
                        numerator = torch.abs(torch.sum(X_ref_frame * torch.conj(X_est_frame)))
                        denominator = torch.sqrt(torch.sum(torch.abs(X_ref_frame)**2) * 
                                               torch.sum(torch.abs(X_est_frame)**2) + eps)
                        
                        frame_coherence = numerator / denominator
                        sci_frames.append(frame_coherence)
                    
                    # Average across frames
                    sci_s = torch.stack(sci_frames).mean()
                    sci_sum += sci_s
                
                sci_perms.append(sci_sum)
            
            sci_values.append(torch.max(torch.stack(sci_perms)))
        else:
            sci_sum = 0
            for s in range(n_sources):
                X_ref = torch.stft(target_b[s], n_fft=n_fft, hop_length=hop_length, window=window, 
                                return_complex=True)
                X_est = torch.stft(est_b[s], n_fft=n_fft, hop_length=hop_length, window=window, 
                                return_complex=True)
                
                num_frames = X_ref.shape[1]
                sci_frames = []
                
                for t in range(num_frames):
                    X_ref_frame = X_ref[:, t]
                    X_est_frame = X_est[:, t]
                    
                    numerator = torch.abs(torch.sum(X_ref_frame * torch.conj(X_est_frame)))
                    denominator = torch.sqrt(torch.sum(torch.abs(X_ref_frame)**2) * 
                                           torch.sum(torch.abs(X_est_frame)**2) + eps)
                    
                    frame_coherence = numerator / denominator
                    sci_frames.append(frame_coherence)
                
                sci_s = torch.stack(sci_frames).mean()
                sci_sum += sci_s
            
            sci_values.append(sci_sum / n_sources)  # Average across sources
    
    # Average across batches
    sci = torch.stack(sci_values).mean()
    
    return sci

def compute_pesq(est_sources, target_sources, sample_rate=16000):
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape
    
    est_np = est_sources.detach().cpu().numpy()
    target_np = target_sources.detach().cpu().numpy()
    
    # Calculate PESQ for each source in each batch
    pesq_scores = []
    for b in range(batch_size):
        for s in range(n_sources):
            # Skip silent segments
            if np.max(np.abs(target_np[b, s])) < 1e-6 or np.max(np.abs(est_np[b, s])) < 1e-6:
                continue
                
            try:
                # PESQ requires specific amplitude scaling
                ref_scaled = np.clip(target_np[b, s], -1, 1) * 32768
                est_scaled = np.clip(est_np[b, s], -1, 1) * 32768
                
                # Calculate PESQ (wb = wideband)
                score = pesq(sample_rate, ref_scaled, est_scaled, 'wb')
                pesq_scores.append(score)
            except Exception as e:
                print(f"PESQ calculation error: {e}")
    
    # Return average PESQ
    if pesq_scores:
        return sum(pesq_scores) / len(pesq_scores)
    else:
        return float('nan')  # Return NaN if no valid scores

def compute_stoi(est_sources, target_sources, sample_rate=16000):
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape
    
    est_np = est_sources.detach().cpu().numpy()
    target_np = target_sources.detach().cpu().numpy()
    
    # Calculate STOI for each source in each batch
    stoi_scores = []
    for b in range(batch_size):
        for s in range(n_sources):
            # Skip silent segments
            if np.max(np.abs(target_np[b, s])) < 1e-6 or np.max(np.abs(est_np[b, s])) < 1e-6:
                continue
                
            try:
                # Calculate STOI
                score = stoi(target_np[b, s], est_np[b, s], sample_rate, extended=False)
                stoi_scores.append(score)
            except Exception as e:
                print(f"STOI calculation error: {e}")
    
    # Return average STOI
    if stoi_scores:
        return sum(stoi_scores) / len(stoi_scores)
    else:
        return float('nan')  # Return NaN if no valid scores

def evaluate_model(model, test_loader, device, epoch=None, save_dir=None):
    """Evaluate the model on the test set with multiple metrics"""
    model.eval()
    
    # Initialize metric accumulators
    metrics = {
        'si_sdr': 0.0,
        'sci': 0.0,
        'pesq': [],
        'stoi': 0.0,
        'count': 0
    }
    
    with torch.no_grad():
        for batch_idx, (mixture, sources) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            # SepFormer expects [B, T] input format
            if mixture.dim() == 3 and mixture.shape[1] == 1:
                mixture = mixture.squeeze(1)
            
            # Use the model's separate_batch method
            est_sources = model.separate_batch(mixture)
            
            # Model returns [B, T, num_sources], permute to [B, num_sources, T]
            est_sources = est_sources.permute(0, 2, 1)
            
            # Adjust number of sources if needed
            if est_sources.shape[1] != sources.shape[1]:
                est_sources = est_sources[:, :sources.shape[1], :]
            
            # Apply normalization to the estimated sources
            est_sources = normalize_sources(est_sources)
            
            # Calculate SI-SDR
            si_sdr = si_sdr_loss(est_sources, sources)
            metrics['si_sdr'] += si_sdr.item() * mixture.size(0)
            
            # Calculate SCI
            sci = spectral_coherence_index_loss(est_sources, sources)
            metrics['sci'] += sci.item() * mixture.size(0)
            
            # Calculate STOI
            stoi_score = compute_stoi(est_sources, sources)
            if not math.isnan(stoi_score):
                metrics['stoi'] += stoi_score * mixture.size(0)
            
            # Calculate PESQ (handled separately due to potential NaN values)
            pesq_score = compute_pesq(est_sources, sources)
            if not math.isnan(pesq_score):
                metrics['pesq'].append(pesq_score)
            
            metrics['count'] += mixture.size(0)
            
            # Save visualization for first batch only
            if save_dir and batch_idx == 0:
                save_visualization(mixture[0], sources[0], est_sources[0], 
                                save_dir, epoch, "sample_1")
    
    # Average the metrics
    results = {
        'si_sdr': metrics['si_sdr'] / metrics['count'],
        'sci': metrics['sci'] / metrics['count'],
        'stoi': metrics['stoi'] / metrics['count']
    }
    
    # Handle PESQ averaging separately
    if metrics['pesq']:
        results['pesq'] = sum(metrics['pesq']) / len(metrics['pesq'])
    else:
        results['pesq'] = float('nan')
    
    model.train()
    return results

def save_visualization(mixture, sources, est_sources, save_dir, epoch, name):
    total_plots = 1 + sources.shape[0] + est_sources.shape[0]
    
    fig, axes = plt.subplots(total_plots, 1, figsize=(15, 3*total_plots))
    
    # Plot mixture
    mixture_cpu = mixture.cpu().numpy()
    axes[0].plot(mixture_cpu)
    axes[0].set_title("Mixture")
    axes[0].set_ylim(-1.1, 1.1)
    
    # Plot ground truth sources
    for i in range(sources.shape[0]):
        source_cpu = sources[i].cpu().numpy()
        axes[i+1].plot(source_cpu)
        axes[i+1].set_title(f"Source {i+1} (Ground Truth)")
        axes[i+1].set_ylim(-1.1, 1.1)
    
    # Plot estimated sources
    for i in range(est_sources.shape[0]):
        est_source_cpu = est_sources[i].cpu().numpy()
        axes[i+1+sources.shape[0]].plot(est_source_cpu)
        axes[i+1+sources.shape[0]].set_title(f"Source {i+1} (Estimated)")
        axes[i+1+sources.shape[0]].set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    epoch_str = f"epoch_{epoch}_" if epoch is not None else ""
    plt.savefig(os.path.join(save_dir, f"{epoch_str}{name}.png"))
    plt.close()

def plot_spectrogram(waveform, sample_rate, title="Spectrogram", save_path=None):
    """Plot and optionally save a spectrogram"""
    waveform_np = waveform.cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    spec = librosa.feature.melspectrogram(y=waveform_np, sr=sample_rate)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    librosa.display.specshow(spec_db, sr=sample_rate, 
                           x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def finetune_sepformer_with_lora(audio_folder, output_dir, num_sources=3, epochs=10, 
                               batch_size=1, eval_interval=2, learning_rate=1e-4,
                               lora_rank=4, lora_alpha=16, loss_type='si_sdr'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training with loss type: {loss_type}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Set up HuggingFace authentication
    setup_huggingface_auth()
    
    print("Loading pre-trained SepFormer model...")
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-libri3mix", 
        savedir="pretrained_models/sepformer-libri3mix",
        run_opts={"device": device}
    )
    
    # Move model to device
    model = model.to(device)
    
    # Apply LoRA to MultiheadAttention modules
    model, replaced_modules = apply_lora_to_multihead_attention(
        model, 
        rank=lora_rank, 
        alpha=lora_alpha
    )
    
    # Prepare dataset
    print("Preparing dataset...")
    full_dataset = MixedAudioDataset(audio_folder, num_sources=num_sources)
    
    # Split into train and test sets (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set trainable parameters (only LoRA parameters)
    trainable_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ["lora_A", "lora_B", "q_lora", "k_lora", "v_lora", "out_lora"]):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Initialize optimizer with only trainable parameters
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
    # Evaluate pre-trained model
    print("Evaluating pre-trained model...")
    pretrained_metrics = evaluate_model(model, test_loader, device, save_dir=vis_dir)
    print(f"Pre-trained model SI-SDR: {pretrained_metrics['si_sdr']:.4f} dB")
    print(f"Pre-trained model SCI: {pretrained_metrics['sci']:.4f}")
    print(f"Pre-trained model PESQ: {pretrained_metrics['pesq']:.4f}")
    print(f"Pre-trained model STOI: {pretrained_metrics['stoi']:.4f}")
    
    # Training loop
    model.train()
    metrics = {
        'epoch': [],
        'train_loss': [],
        'eval_si_sdr': [],
        'eval_sci': [],
        'eval_pesq': [],
        'eval_stoi': [],
        'eval_epochs': []
    }
    
    best_metric = pretrained_metrics['si_sdr'] if loss_type == 'si_sdr' else pretrained_metrics['sci']
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        running_loss = 0.0
        
        # Training
        for i, (mixture, sources) in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}")):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            # Ensure mixture requires grad
            mixture = mixture.clone().detach().requires_grad_(True)
            
            if mixture.dim() == 3 and mixture.shape[1] == 1:
                mixture = mixture.squeeze(1)
                
            # Forward pass
            est_sources = model.separate_batch(mixture)
            
            # Handle the output shape: model returns [B, T, num_sources]
            # We need [B, num_sources, T]
            est_sources = est_sources.permute(0, 2, 1)
            
            # Adjust number of sources if needed
            if est_sources.shape[1] != sources.shape[1]:
                est_sources = est_sources[:, :sources.shape[1], :]
            
            # Apply normalization to the estimated sources
            est_sources = normalize_sources(est_sources)
            
            # Loss based on chosen loss type
            if loss_type == 'si_sdr':
                # SI-SDR Loss (negative because we want to maximize SDR)
                loss = -si_sdr_loss(est_sources, sources)
            elif loss_type == 'sci':
                # SCI Loss (negative because we want to maximize coherence)
                loss = -spectral_coherence_index_loss(est_sources, sources)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            # Check if loss requires grad
            if not loss.requires_grad:
                print("Warning: Loss does not require grad!")
                # Try to reconnect the gradient
                loss = loss.clone().detach().requires_grad_(True)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average training loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(avg_loss)
        
        # Evaluate and save model at specified intervals
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            # Evaluate
            current_metrics = evaluate_model(model, test_loader, device, 
                                          epoch=epoch+1, save_dir=vis_dir)
            print(f"Epoch {epoch+1}, Evaluation SI-SDR: {current_metrics['si_sdr']:.4f} dB")
            print(f"Epoch {epoch+1}, Evaluation SCI: {current_metrics['sci']:.4f}")
            print(f"Epoch {epoch+1}, Evaluation PESQ: {current_metrics['pesq']:.4f}")
            print(f"Epoch {epoch+1}, Evaluation STOI: {current_metrics['stoi']:.4f}")
            
            metrics['eval_si_sdr'].append(current_metrics['si_sdr'])
            metrics['eval_sci'].append(current_metrics['sci'])
            metrics['eval_pesq'].append(current_metrics['pesq'])
            metrics['eval_stoi'].append(current_metrics['stoi'])
            metrics['eval_epochs'].append(epoch + 1)
            
            # Save checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"sepformer_lora_{loss_type}_epoch_{epoch+1}.pt")
            # Get state dict for LoRA layers only to save space
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
            torch.save({
                'epoch': epoch + 1,
                'lora_state_dict': lora_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': current_metrics
            }, ckpt_path)
            
            # Determine best model based on the chosen loss type
            current_metric = current_metrics['si_sdr'] if loss_type == 'si_sdr' else current_metrics['sci']
            if current_metric > best_metric:
                best_metric = current_metric
                best_ckpt_path = os.path.join(ckpt_dir, f"sepformer_lora_{loss_type}_best.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'lora_state_dict': lora_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'metrics': current_metrics
                }, best_ckpt_path)
                print(f"New best model saved with {loss_type.upper()}: {current_metric:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': metrics['epoch'],
        'train_loss': metrics['train_loss'],
    })
    eval_df = pd.DataFrame({
        'epoch': metrics['eval_epochs'],
        'si_sdr': metrics['eval_si_sdr'],
        'sci': metrics['eval_sci'],
        'pesq': metrics['eval_pesq'],
        'stoi': metrics['eval_stoi']
    })
    metrics_df.to_csv(os.path.join(output_dir, f"training_metrics_{loss_type}.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, f"evaluation_metrics_{loss_type}.csv"), index=False)
    
    # Plot metrics
    plot_training_metrics(metrics, pretrained_metrics['si_sdr'], output_dir, loss_type)
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_model(model, test_loader, device, epoch="final", save_dir=vis_dir)
    print(f"Final model SI-SDR: {final_metrics['si_sdr']:.4f} dB")
    print(f"Final model SCI: {final_metrics['sci']:.4f}")
    print(f"Final model PESQ: {final_metrics['pesq']:.4f}")
    print(f"Final model STOI: {final_metrics['stoi']:.4f}")
    
    # Save comparison results
    with open(os.path.join(output_dir, f"results_summary_{loss_type}.txt"), 'w') as f:
        f.write(f"Loss type: {loss_type}\n")
        f.write(f"LoRA Parameters: rank={lora_rank}, alpha={lora_alpha}\n\n")
        f.write(f"Pre-trained model SI-SDR: {pretrained_metrics['si_sdr']:.4f} dB\n")
        f.write(f"Fine-tuned model SI-SDR: {final_metrics['si_sdr']:.4f} dB\n")
        f.write(f"SI-SDR Improvement: {final_metrics['si_sdr'] - pretrained_metrics['si_sdr']:.4f} dB\n\n")
        f.write(f"Pre-trained model SCI: {pretrained_metrics['sci']:.4f}\n")
        f.write(f"Fine-tuned model SCI: {final_metrics['sci']:.4f}\n")
        f.write(f"SCI Improvement: {final_metrics['sci'] - pretrained_metrics['sci']:.4f}\n\n")
        f.write(f"Pre-trained model PESQ: {pretrained_metrics['pesq']:.4f}\n")
        f.write(f"Fine-tuned model PESQ: {final_metrics['pesq']:.4f}\n")
        f.write(f"PESQ Improvement: {final_metrics['pesq'] - pretrained_metrics['pesq']:.4f}\n\n")
        f.write(f"Pre-trained model STOI: {pretrained_metrics['stoi']:.4f}\n")
        f.write(f"Fine-tuned model STOI: {final_metrics['stoi']:.4f}\n")
        f.write(f"STOI Improvement: {final_metrics['stoi'] - pretrained_metrics['stoi']:.4f}\n")
    
    return model, final_metrics, pretrained_metrics

def plot_training_metrics(metrics, pretrained_si_sdr, output_dir, loss_type='si_sdr'):
    """Plot training loss and evaluation metrics"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training loss - create DataFrame with just training data
    train_df = pd.DataFrame({
        'epoch': metrics['epoch'],
        'train_loss': metrics['train_loss']
    })
    sns.lineplot(x="epoch", y="train_loss", data=train_df, ax=axes[0, 0], marker='o')
    axes[0, 0].set_title(f'Training Loss ({loss_type.upper()})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Plot evaluation SI-SDR if available
    if metrics['eval_si_sdr']:
        eval_df = pd.DataFrame({
            'epoch': metrics['eval_epochs'],
            'si_sdr': metrics['eval_si_sdr']
        })
        
        sns.lineplot(x="epoch", y="si_sdr", data=eval_df, ax=axes[0, 1], marker='o')
        
        # Add horizontal line for pretrained SI-SDR
        axes[0, 1].axhline(y=pretrained_si_sdr, color='r', linestyle='--', 
                  label=f'Pre-trained: {pretrained_si_sdr:.2f} dB')
        axes[0, 1].legend()
        
        axes[0, 1].set_title('Evaluation SI-SDR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SI-SDR (dB)')
    
    # Plot evaluation SCI if available
    if metrics['eval_sci']:
        eval_sci_df = pd.DataFrame({
            'epoch': metrics['eval_epochs'],
            'sci': metrics['eval_sci']
        })
        
        sns.lineplot(x="epoch", y="sci", data=eval_sci_df, ax=axes[1, 0], marker='o')
        
        # Add horizontal line for pretrained SCI
        if len(metrics['eval_sci']) > 0:
            pretrained_sci = metrics['eval_sci'][0]
            axes[1, 0].axhline(y=pretrained_sci, color='r', linestyle='--', 
                      label=f'Pre-trained: {pretrained_sci:.2f}')
            axes[1, 0].legend()
        
        axes[1, 0].set_title('Evaluation SCI')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SCI')
    
    # Plot evaluation PESQ if available
    if metrics['eval_pesq']:
        eval_pesq_df = pd.DataFrame({
            'epoch': metrics['eval_epochs'],
            'pesq': metrics['eval_pesq']
        })
        
        sns.lineplot(x="epoch", y="pesq", data=eval_pesq_df, ax=axes[1, 1], marker='o')
        
        # Add horizontal line for pretrained PESQ
        if len(metrics['eval_pesq']) > 0:
            pretrained_pesq = metrics['eval_pesq'][0]
            axes[1, 1].axhline(y=pretrained_pesq, color='r', linestyle='--', 
                      label=f'Pre-trained: {pretrained_pesq:.2f}')
            axes[1, 1].legend()
        
        axes[1, 1].set_title('Evaluation PESQ')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('PESQ')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_metrics_{loss_type}.png"))
    plt.close()
    
    # Create metric comparison bar chart for final results
    if metrics['eval_si_sdr'] and metrics['eval_sci'] and metrics['eval_pesq'] and metrics['eval_stoi']:
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SI-SDR comparison
        bars1 = axes[0, 0].bar(['Pre-trained', 'Fine-tuned'], 
                              [pretrained_si_sdr, metrics['eval_si_sdr'][-1]], 
                              color=['#1f77b4', '#2ca02c'])
        axes[0, 0].set_title('SI-SDR Comparison')
        axes[0, 0].set_ylabel('SI-SDR (dB)')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        for bar, val in zip(bars1, [pretrained_si_sdr, metrics['eval_si_sdr'][-1]]):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.1, 
                    f'{val:.2f} dB', ha='center', va='bottom')
        
        # SCI comparison
        bars2 = axes[0, 1].bar(['Pre-trained', 'Fine-tuned'], 
                              [metrics['eval_sci'][0], metrics['eval_sci'][-1]], 
                              color=['#1f77b4', '#2ca02c'])
        axes[0, 1].set_title('SCI Comparison')
        axes[0, 1].set_ylabel('SCI')
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        for bar, val in zip(bars2, [metrics['eval_sci'][0], metrics['eval_sci'][-1]]):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom')
        
        # PESQ comparison
        bars3 = axes[1, 0].bar(['Pre-trained', 'Fine-tuned'], 
                              [metrics['eval_pesq'][0], metrics['eval_pesq'][-1]], 
                              color=['#1f77b4', '#2ca02c'])
        axes[1, 0].set_title('PESQ Comparison')
        axes[1, 0].set_ylabel('PESQ')
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        for bar, val in zip(bars3, [metrics['eval_pesq'][0], metrics['eval_pesq'][-1]]):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom')
        
        # STOI comparison
        bars4 = axes[1, 1].bar(['Pre-trained', 'Fine-tuned'], 
                              [metrics['eval_stoi'][0], metrics['eval_stoi'][-1]], 
                              color=['#1f77b4', '#2ca02c'])
        axes[1, 1].set_title('STOI Comparison')
        axes[1, 1].set_ylabel('STOI')
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        for bar, val in zip(bars4, [metrics['eval_stoi'][0], metrics['eval_stoi'][-1]]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_comparison_{loss_type}.png"))
        plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    audio_folder = "/home/m23csa001/Aditi_Speech/libri/audio"  # Path to audio files
    base_output_dir = "lora_sepformer_results"
    num_sources = 3  # Number of sources to mix
    epochs = 6
    batch_size = 4
    eval_interval = 2  # Evaluate every n epochs
    learning_rate = 1e-4
    
    # LoRA parameters
    lora_rank = 4
    lora_alpha = 16
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Run fine-tuning with SI-SDR loss
    print("\n===== Fine-tuning with SI-SDR loss =====")
    output_dir_si_sdr = os.path.join(base_output_dir, "si_sdr_loss")
    model_si_sdr, final_metrics_si_sdr, pretrained_metrics = finetune_sepformer_with_lora(
        audio_folder=audio_folder,
        output_dir=output_dir_si_sdr,
        num_sources=num_sources,
        epochs=epochs,
        batch_size=batch_size,
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        loss_type='si_sdr'
    )
    
    # Run fine-tuning with SCI loss
    print("\n===== Fine-tuning with SCI loss =====")
    output_dir_sci = os.path.join(base_output_dir, "sci_loss")
    model_sci, final_metrics_sci, _ = finetune_sepformer_with_lora(
        audio_folder=audio_folder,
        output_dir=output_dir_sci,
        num_sources=num_sources,
        epochs=epochs,
        batch_size=batch_size,
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        loss_type='sci'
    )
    
    # Create a comprehensive comparison report
    with open(os.path.join(base_output_dir, "comparison_summary.txt"), 'w') as f:
        f.write("==== Audio Source Separation with LoRA Fine-tuning Comparison ====\n\n")
        f.write(f"LoRA Parameters: rank={lora_rank}, alpha={lora_alpha}\n")
        f.write(f"Training epochs: {epochs}\n\n")
        
        f.write("--- Pretrained Model Metrics ---\n")
        f.write(f"SI-SDR: {pretrained_metrics['si_sdr']:.4f} dB\n")
        f.write(f"SCI: {pretrained_metrics['sci']:.4f}\n")
        f.write(f"PESQ: {pretrained_metrics['pesq']:.4f}\n")
        f.write(f"STOI: {pretrained_metrics['stoi']:.4f}\n\n")
        
        f.write("--- SI-SDR Loss Fine-tuning Results ---\n")
        f.write(f"SI-SDR: {final_metrics_si_sdr['si_sdr']:.4f} dB (Improvement: {final_metrics_si_sdr['si_sdr'] - pretrained_metrics['si_sdr']:.4f} dB)\n")
        f.write(f"SCI: {final_metrics_si_sdr['sci']:.4f} (Change: {final_metrics_si_sdr['sci'] - pretrained_metrics['sci']:.4f})\n")
        f.write(f"PESQ: {final_metrics_si_sdr['pesq']:.4f} (Change: {final_metrics_si_sdr['pesq'] - pretrained_metrics['pesq']:.4f})\n")
        f.write(f"STOI: {final_metrics_si_sdr['stoi']:.4f} (Change: {final_metrics_si_sdr['stoi'] - pretrained_metrics['stoi']:.4f})\n\n")
        
        f.write("--- SCI Loss Fine-tuning Results ---\n")
        f.write(f"SI-SDR: {final_metrics_sci['si_sdr']:.4f} dB (Change: {final_metrics_sci['si_sdr'] - pretrained_metrics['si_sdr']:.4f} dB)\n")
        f.write(f"SCI: {final_metrics_sci['sci']:.4f} (Improvement: {final_metrics_sci['sci'] - pretrained_metrics['sci']:.4f})\n")
        f.write(f"PESQ: {final_metrics_sci['pesq']:.4f} (Change: {final_metrics_sci['pesq'] - pretrained_metrics['pesq']:.4f})\n")
        f.write(f"STOI: {final_metrics_sci['stoi']:.4f} (Change: {final_metrics_sci['stoi'] - pretrained_metrics['stoi']:.4f})\n\n")
        
        f.write("--- Comparison Summary ---\n")
        f.write("Best SI-SDR: " + ("SI-SDR loss" if final_metrics_si_sdr['si_sdr'] > final_metrics_sci['si_sdr'] else "SCI loss") + "\n")
        f.write("Best SCI: " + ("SI-SDR loss" if final_metrics_si_sdr['sci'] > final_metrics_sci['sci'] else "SCI loss") + "\n")
        f.write("Best PESQ: " + ("SI-SDR loss" if final_metrics_si_sdr['pesq'] > final_metrics_sci['pesq'] else "SCI loss") + "\n")
        f.write("Best STOI: " + ("SI-SDR loss" if final_metrics_si_sdr['stoi'] > final_metrics_sci['stoi'] else "SCI loss") + "\n")
    
    # Create comparison plots
    create_comparison_plots(pretrained_metrics, final_metrics_si_sdr, final_metrics_sci, base_output_dir)
    
    print("\nFine-tuning complete! See comparison results in:", os.path.join(base_output_dir, "comparison_summary.txt"))

def create_comparison_plots(pretrained_metrics, si_sdr_metrics, sci_metrics, output_dir):
    """Create comparative visualizations of all three models"""
    # Set style
    sns.set_style("whitegrid")
    
    # Create bar plot for all metrics
    metrics_names = ['SI-SDR (dB)', 'SCI', 'PESQ', 'STOI']
    model_names = ['Pretrained', 'SI-SDR Loss', 'SCI Loss']
    
    # Extract metric values
    si_sdr_values = [pretrained_metrics['si_sdr'], si_sdr_metrics['si_sdr'], sci_metrics['si_sdr']]
    sci_values = [pretrained_metrics['sci'], si_sdr_metrics['sci'], sci_metrics['sci']]
    pesq_values = [pretrained_metrics['pesq'], si_sdr_metrics['pesq'], sci_metrics['pesq']]
    stoi_values = [pretrained_metrics['stoi'], si_sdr_metrics['stoi'], sci_metrics['stoi']]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # SI-SDR plot
    axes[0, 0].bar(model_names, si_sdr_values, color=['#1f77b4', '#2ca02c', '#d62728'])
    axes[0, 0].set_title('SI-SDR Comparison', fontsize=14)
    axes[0, 0].set_ylabel('SI-SDR (dB)', fontsize=12)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(si_sdr_values):
        axes[0, 0].text(i, val + 0.1, f'{val:.2f}', ha='center', fontsize=10)
    
    # SCI plot
    axes[0, 1].bar(model_names, sci_values, color=['#1f77b4', '#2ca02c', '#d62728'])
    axes[0, 1].set_title('SCI Comparison', fontsize=14)
    axes[0, 1].set_ylabel('SCI', fontsize=12)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(sci_values):
        axes[0, 1].text(i, val + 0.01, f'{val:.2f}', ha='center', fontsize=10)
    
    # PESQ plot
    axes[1, 0].bar(model_names, pesq_values, color=['#1f77b4', '#2ca02c', '#d62728'])
    axes[1, 0].set_title('PESQ Comparison', fontsize=14)
    axes[1, 0].set_ylabel('PESQ', fontsize=12)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(pesq_values):
        axes[1, 0].text(i, val + 0.01, f'{val:.2f}', ha='center', fontsize=10)
    
    # STOI plot
    axes[1, 1].bar(model_names, stoi_values, color=['#1f77b4', '#2ca02c', '#d62728'])
    axes[1, 1].set_title('STOI Comparison', fontsize=14)
    axes[1, 1].set_ylabel('STOI', fontsize=12)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, val in enumerate(stoi_values):
        axes[1, 1].text(i, val + 0.01, f'{val:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_metrics_comparison.png"))
    plt.close()
    
    # Create relative improvement plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate relative improvements
    si_sdr_improvement = [(si_sdr_metrics['si_sdr'] - pretrained_metrics['si_sdr']), 
                          (sci_metrics['si_sdr'] - pretrained_metrics['si_sdr'])]
    sci_improvement = [(si_sdr_metrics['sci'] - pretrained_metrics['sci']), 
                       (sci_metrics['sci'] - pretrained_metrics['sci'])]
    pesq_improvement = [(si_sdr_metrics['pesq'] - pretrained_metrics['pesq']), 
                        (sci_metrics['pesq'] - pretrained_metrics['pesq'])]
    stoi_improvement = [(si_sdr_metrics['stoi'] - pretrained_metrics['stoi']), 
                        (sci_metrics['stoi'] - pretrained_metrics['stoi'])]
    
    # Bar width
    width = 0.2
    x = np.arange(2)  # Two fine-tuning methods
    
    # Plot improvements
    axes[0].bar(x - 1.5*width, si_sdr_improvement, width, label='SI-SDR (dB)')
    axes[0].bar(x - 0.5*width, sci_improvement, width, label='SCI')
    axes[0].bar(x + 0.5*width, pesq_improvement, width, label='PESQ')
    axes[0].bar(x + 1.5*width, stoi_improvement, width, label='STOI')
    
    axes[0].set_title('Absolute Metric Improvements', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['SI-SDR Loss', 'SCI Loss'])
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate percentage improvements
    si_sdr_pct = [(si_sdr_metrics['si_sdr'] / pretrained_metrics['si_sdr'] - 1) * 100, 
                  (sci_metrics['si_sdr'] / pretrained_metrics['si_sdr'] - 1) * 100]
    sci_pct = [(si_sdr_metrics['sci'] / pretrained_metrics['sci'] - 1) * 100, 
               (sci_metrics['sci'] / pretrained_metrics['sci'] - 1) * 100]
    pesq_pct = [(si_sdr_metrics['pesq'] / pretrained_metrics['pesq'] - 1) * 100, 
                (sci_metrics['pesq'] / pretrained_metrics['pesq'] - 1) * 100]
    stoi_pct = [(si_sdr_metrics['stoi'] / pretrained_metrics['stoi'] - 1) * 100, 
                (sci_metrics['stoi'] / pretrained_metrics['stoi'] - 1) * 100]
    
    # Plot percentage improvements
    axes[1].bar(x - 1.5*width, si_sdr_pct, width, label='SI-SDR')
    axes[1].bar(x - 0.5*width, sci_pct, width, label='SCI')
    axes[1].bar(x + 0.5*width, pesq_pct, width, label='PESQ')
    axes[1].bar(x + 1.5*width, stoi_pct, width, label='STOI')
    
    axes[1].set_title('Percentage Metric Improvements (%)', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['SI-SDR Loss', 'SCI Loss'])
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_improvements.png"))
    plt.close()

if __name__ == "__main__":
    main()
