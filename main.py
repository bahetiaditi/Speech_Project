import os
import random
import numpy as np
import torch
import pandas as pd
from speechbrain.pretrained import SepformerSeparation

from data_utils import create_train_test_dataloaders
from train import finetune_sepformer_with_lora

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    audio_folder = "/home/m23csa001/Aditi_Speech/CREMA/CREMA-D" 
    base_output_dir = "lora_sepformer_results"
    num_sources = 3  
    epochs = 10
    batch_size = 4
    eval_interval = 2  
    learning_rate = 1e-4
    segment_length = 4.0
    sample_rate = 16000
    num_workers = 4
    
    # LoRA parameters
    lora_rank = 4
    lora_alpha = 16
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    train_loader, test_loader = create_train_test_dataloaders(
        audio_folder=audio_folder,
        num_sources=num_sources,
        batch_size=batch_size,
        segment_length=segment_length,
        sample_rate=sample_rate,
        test_size=0.2,  
        num_workers=num_workers,
        min_mixtures=100,
        max_mixtures=5000, 
        use_cache=True
    )
    
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
    
    # Print final comparison
    print("\n===== Final Results Comparison =====")
    print(f"Pre-trained model SI-SDR: {pretrained_metrics['si_sdr']:.4f} dB")
    print(f"SI-SDR trained model SI-SDR: {final_metrics_si_sdr['si_sdr']:.4f} dB")
    print(f"SCI trained model SI-SDR: {final_metrics_sci['si_sdr']:.4f} dB")
    
    print(f"Pre-trained model SCI: {pretrained_metrics['sci']:.4f}")
    print(f"SI-SDR trained model SCI: {final_metrics_si_sdr['sci']:.4f}")
    print(f"SCI trained model SCI: {final_metrics_sci['sci']:.4f}")
    
    # Save comparison results
    comparison_df = pd.DataFrame({
        'Model': ['Pre-trained', 'SI-SDR trained', 'SCI trained'],
        'SI-SDR': [pretrained_metrics['si_sdr'], final_metrics_si_sdr['si_sdr'], final_metrics_sci['si_sdr']],
        'SCI': [pretrained_metrics['sci'], final_metrics_si_sdr['sci'], final_metrics_sci['sci']],
        'PESQ': [pretrained_metrics['pesq'], final_metrics_si_sdr['pesq'], final_metrics_sci['pesq']],
        'STOI': [pretrained_metrics['stoi'], final_metrics_si_sdr['stoi'], final_metrics_sci['stoi']]
    })
    
    comparison_df.to_csv(os.path.join(base_output_dir, "model_comparison.csv"), index=False)
    print(f"Comparison saved to {os.path.join(base_output_dir, 'model_comparison.csv')}")

if __name__ == "__main__":
    main()
