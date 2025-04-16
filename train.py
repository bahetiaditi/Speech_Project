import os
import torch
import pandas as pd
from tqdm import tqdm

from lora_module import apply_lora_to_multihead_attention
from evaluation import si_sdr_loss, spectral_coherence_index_loss, normalize_sources, evaluate_model, plot_training_metrics

def finetune_sepformer_with_lora(audio_folder, output_dir, num_sources=3, epochs=10, 
                               batch_size=1, eval_interval=2, learning_rate=1e-4,
                               lora_rank=4, lora_alpha=16, loss_type='si_sdr'):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training with loss type: {loss_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    setup_huggingface_auth()
    
    print("Loading pre-trained SepFormer model...")
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-libri3mix", 
        savedir="pretrained_models/sepformer-libri3mix",
        run_opts={"device": device}
    )
    
    model = model.to(device)
    
    model, replaced_modules = apply_lora_to_multihead_attention(
        model, 
        rank=lora_rank, 
        alpha=lora_alpha
    )
    
    print("Preparing dataset...")
    
    trainable_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ["lora_A", "lora_B", "q_lora", "k_lora", "v_lora", "out_lora"]):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    
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
        
        for i, (mixture, sources) in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}")):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            mixture = mixture.clone().detach().requires_grad_(True)
            
            if mixture.dim() == 3 and mixture.shape[1] == 1:
                mixture = mixture.squeeze(1)
                
            # Forward pass
            est_sources = model.separate_batch(mixture)
            est_sources = est_sources.permute(0, 2, 1)
            
            if est_sources.shape[1] != sources.shape[1]:
                est_sources = est_sources[:, :sources.shape[1], :]
            
            est_sources = normalize_sources(est_sources)
            
            if loss_type == 'si_sdr':
                # SI-SDR Loss (negative because we want to maximize SDR)
                loss = -si_sdr_loss(est_sources, sources)
            elif loss_type == 'sci':
                # SCI Loss (negative because we want to maximize coherence)
                loss = -spectral_coherence_index_loss(est_sources, sources)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            if not loss.requires_grad:
                print("Warning: Loss does not require grad!")
                loss = loss.clone().detach().requires_grad_(True)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(avg_loss)
      
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
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
            torch.save({
                'epoch': epoch + 1,
                'lora_state_dict': lora_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': current_metrics
            }, ckpt_path)
            
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
    
    plot_training_metrics(metrics, pretrained_metrics['si_sdr'], output_dir, loss_type)
    
    print("\nFinal evaluation...")
    final_metrics = evaluate_model(model, test_loader, device, epoch="final", save_dir=vis_dir)
    print(f"Final model SI-SDR: {final_metrics['si_sdr']:.4f} dB")
    print(f"Final model SCI: {final_metrics['sci']:.4f}")
    print(f"Final model PESQ: {final_metrics['pesq']:.4f}")
    print(f"Final model STOI: {final_metrics['stoi']:.4f}")
    
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
