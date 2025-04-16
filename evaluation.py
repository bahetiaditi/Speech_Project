import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import math
from itertools import permutations
from tqdm import tqdm

def si_sdr_loss(est_sources, target_sources, eps=1e-8, permutation_invariant=True):
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape
    
    si_sdrs = []
    for b in range(batch_size):
        est_b = est_sources[b]  # [n_sources, time]
        target_b = target_sources[b]  # [n_sources, time]
        
        est_b = est_b - est_b.mean(dim=1, keepdim=True)
        target_b = target_b - target_b.mean(dim=1, keepdim=True)
        
        if permutation_invariant and n_sources > 1:
            all_perms = list(permutations(range(n_sources)))
            
            # Calculate SDR for each permutation
            sdr_perms = []
            for perm in all_perms:
                est_perm = est_b[list(perm)]
                
                # Calculate SI-SDR for this permutation
                sdr_sum = 0
                for s in range(n_sources):
                    s_target = (torch.sum(est_perm[s] * target_b[s]) * target_b[s]) / (torch.sum(target_b[s]**2) + eps)
                    
                    e_noise = est_perm[s] - s_target
                    
                    si_sdr_s = 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + eps) + eps)
                    sdr_sum += si_sdr_s
                
                sdr_perms.append(sdr_sum)
            
            si_sdrs.append(torch.max(torch.stack(sdr_perms)))
        else:
            # Standard calculation without permutation
            sdr_sum = 0
            for s in range(n_sources):
                # Target projection
                s_target = (torch.sum(est_b[s] * target_b[s]) * target_b[s]) / (torch.sum(target_b[s]**2) + eps)
                
                # Error
                e_noise = est_b[s] - s_target
                
                # SI-SDR for this source
                si_sdr_s = 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + eps) + eps)
                sdr_sum += si_sdr_s
            
            si_sdrs.append(sdr_sum / n_sources) 
    
    si_sdr = torch.stack(si_sdrs).mean()
    
    return si_sdr

def normalize_sources(est_sources, max_amplitude=0.9):
    """Simple amplitude normalization for estimated sources"""
    
    normalized = est_sources.clone()
    max_vals = normalized.abs().max(dim=-1, keepdim=True)[0]
    max_vals = torch.clamp(max_vals, min=1e-8)
    normalized = normalized * (max_amplitude / max_vals)
    
    return normalized

def spectral_coherence_index_loss(est_sources, target_sources, n_fft=512, hop_length=128, eps=1e-8, permutation_invariant=True):
    # Ensure input shapes: [batch, n_sources, time]
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape
    
    # Window function
    window = torch.hann_window(n_fft).to(est_sources.device)
    
    # For each batch element
    sci_values = []
    for b in range(batch_size):
        est_b = est_sources[b]  # [n_sources, time]
        target_b = target_sources[b]  # [n_sources, time]
        
        if permutation_invariant and n_sources > 1:
            # Try all possible permutations of sources
            all_perms = list(permutations(range(n_sources)))
            
            # Calculate SCI for each permutation
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
            
            # Use the permutation with highest SCI
            sci_values.append(torch.max(torch.stack(sci_perms)))
        else:
            # Standard calculation without permutation
            sci_sum = 0
            for s in range(n_sources):
                # Compute STFTs
                X_ref = torch.stft(target_b[s], n_fft=n_fft, hop_length=hop_length, window=window, 
                                return_complex=True)
                X_est = torch.stft(est_b[s], n_fft=n_fft, hop_length=hop_length, window=window, 
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
            
            sci_values.append(sci_sum / n_sources) 
    
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
    
    pesq_scores = []
    for b in range(batch_size):
        for s in range(n_sources):
            if np.max(np.abs(target_np[b, s])) < 1e-6 or np.max(np.abs(est_np[b, s])) < 1e-6:
                continue
                
            try:
                ref_scaled = np.clip(target_np[b, s], -1, 1) * 32768
                est_scaled = np.clip(est_np[b, s], -1, 1) * 32768
              
                score = pesq(sample_rate, ref_scaled, est_scaled, 'wb')
                pesq_scores.append(score)
            except Exception as e:
                print(f"PESQ calculation error: {e}")
    
    if pesq_scores:
        return sum(pesq_scores) / len(pesq_scores)
    else:
        return float('nan')  

def compute_stoi(est_sources, target_sources, sample_rate=16000):
    if est_sources.dim() == 2:
        est_sources = est_sources.unsqueeze(0)
    if target_sources.dim() == 2:
        target_sources = target_sources.unsqueeze(0)
    
    batch_size, n_sources, length = est_sources.shape

    est_np = est_sources.detach().cpu().numpy()
    target_np = target_sources.detach().cpu().numpy()
    
    stoi_scores = []
    for b in range(batch_size):
        for s in range(n_sources):
            if np.max(np.abs(target_np[b, s])) < 1e-6 or np.max(np.abs(est_np[b, s])) < 1e-6:
                continue
                
            try:
                score = stoi(target_np[b, s], est_np[b, s], sample_rate, extended=False)
                stoi_scores.append(score)
            except Exception as e:
                print(f"STOI calculation error: {e}")
    
    if stoi_scores:
        return sum(stoi_scores) / len(stoi_scores)
    else:
        return float('nan')  

def evaluate_model(model, test_loader, device, epoch=None, save_dir=None):
    model.eval()
    
    metrics = {
        'si_sdr': 0.0,
        'sci': 0.0,
        'pesq': [],
        'stoi': 0.0,
        'count': 0
    }
    
    with torch.no_grad():
        for batch_idx, (mixture, sources) in enumerate(tqdm(test_loader, desc="Evaluating")):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            # SepFormer expects [B, T] input format
            if mixture.dim() == 3 and mixture.shape[1] == 1:
                mixture = mixture.squeeze(1)
            
            # Use the model's separate_batch method
            est_sources = model.separate_batch(mixture)
            
            # Model returns [B, T, num_sources], permute to [B, num_sources, T]
            est_sources = est_sources.permute(0, 2, 1)
            
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
            
            # Calculate PESQ 
            pesq_score = compute_pesq(est_sources, sources)
            if not math.isnan(pesq_score):
                metrics['pesq'].append(pesq_score)
            
            metrics['count'] += mixture.size(0)
            
            if save_dir and batch_idx == 0:
                save_visualization(mixture[0], sources[0], est_sources[0], 
                                save_dir, epoch, "sample_1")
    
    results = {
        'si_sdr': metrics['si_sdr'] / metrics['count'],
        'sci': metrics['sci'] / metrics['count'],
        'stoi': metrics['stoi'] / metrics['count']
    }
    
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
      
def plot_training_metrics(metrics, pretrained_si_sdr, output_dir, loss_type):
    """Plot training and evaluation metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Plot SI-SDR
    plt.subplot(2, 2, 2)
    plt.plot(metrics['eval_epochs'], metrics['eval_si_sdr'], 'g-')
    plt.axhline(y=pretrained_si_sdr, color='r', linestyle='--', label='Pre-trained')
    plt.xlabel('Epoch')
    plt.ylabel('SI-SDR (dB)')
    plt.title('SI-SDR Improvement')
    plt.legend()
    plt.grid(True)
    
    # Plot SCI
    plt.subplot(2, 2, 3)
    plt.plot(metrics['eval_epochs'], metrics['eval_sci'], 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('SCI')
    plt.title('Spectral Coherence Index')
    plt.grid(True)
    
    # Plot PESQ and STOI
    plt.subplot(2, 2, 4)
    plt.plot(metrics['eval_epochs'], metrics['eval_pesq'], 'y-', label='PESQ')
    plt.plot(metrics['eval_epochs'], metrics['eval_stoi'], 'c-', label='STOI')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('PESQ and STOI Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_metrics_{loss_type}.png"))
    plt.close()
