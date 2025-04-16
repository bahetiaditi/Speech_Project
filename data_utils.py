import os
import random
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

def create_train_test_dataloaders(audio_folder, num_sources=3, batch_size=16, segment_length=4.0, 
                                  sample_rate=16000, test_size=0.2, num_workers=4, 
                                  min_mixtures=250, max_mixtures=None, use_cache=True):
    audio_files = [os.path.join(audio_folder, f)
                  for f in os.listdir(audio_folder)
                  if f.endswith('.wav')]
    
    valid_files = []
    for file_path in audio_files:
        try:
            info = torchaudio.info(file_path)
            min_length = int(segment_length * sample_rate * 0.8)
            if info.num_frames >= min_length:
                valid_files.append(file_path)
        except Exception as e:
            print(f"Warning: Could not get info for {file_path}: {e}")
    
    if len(valid_files) < num_sources*2: 
        raise ValueError(f"Need at least {num_sources*2} valid audio files, found {len(valid_files)}")
    
    random.shuffle(valid_files)
    
    num_test_files = max(int(len(valid_files) * test_size), num_sources)
    num_train_files = len(valid_files) - num_test_files
    
    train_files = valid_files[:num_train_files]
    test_files = valid_files[num_train_files:]
    
    print(f"Using {len(train_files)} files for training and {len(test_files)} files for testing")
    
    train_max_mixtures = None
    test_max_mixtures = None
    
    if max_mixtures is not None:
        train_ratio = 1 - test_size
        train_max_mixtures = int(max_mixtures * train_ratio)
        test_max_mixtures = max_mixtures - train_max_mixtures
    
    train_dataset = MixedAudioDataset(
        audio_files=train_files,
        num_sources=num_sources,
        segment_length=segment_length,
        sample_rate=sample_rate,
        normalize_method='rms',
        augment=True,
        min_snr=0,
        max_snr=5,
        min_mixtures=min_mixtures,
        max_mixtures=train_max_mixtures,
        use_cache=use_cache
    )
    
    test_dataset = MixedAudioDataset(
        audio_files=test_files,
        num_sources=num_sources,
        segment_length=segment_length,
        sample_rate=sample_rate,
        normalize_method='rms',
        augment=False,
        min_snr=3,  # Fixed SNR for consistent evaluation
        max_snr=3,
        min_mixtures=int(min_mixtures * test_size),
        max_mixtures=test_max_mixtures,
        use_cache=use_cache
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MixedAudioDataset.collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MixedAudioDataset.collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

class MixedAudioDataset(Dataset):
    def __init__(self, audio_files, num_sources=3, segment_length=4.0, sample_rate=16000, 
                 normalize_method='rms', augment=True, min_snr=0, max_snr=5,
                 min_mixtures=250, max_mixtures=None, use_cache=True):

        self.audio_files = audio_files  
        self.num_sources = num_sources
        self.segment_samples = int(segment_length * sample_rate)
        self.sample_rate = sample_rate
        self.normalize_method = normalize_method
        self.augment = augment
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.min_mixtures = min_mixtures
        self.max_mixtures = max_mixtures
        self.use_cache = use_cache
        
        self.file_lengths = {}
        self.audio_cache = {} if use_cache else None
        
        for file_path in self.audio_files:
            try:
                info = torchaudio.info(file_path)
                self.file_lengths[file_path] = info.num_frames
                
                if use_cache:
                    waveform, sr = torchaudio.load(file_path)
                    if sr != sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    self.audio_cache[file_path] = waveform
            except Exception as e:
                print(f"Warning: Could not get info for {file_path}: {e}")
        
        import math
        from itertools import combinations
        
        num_files = len(self.audio_files)
        if num_files < num_sources:
            raise ValueError(f"Need at least {num_sources} valid audio files, found {num_files}")
            
        max_combinations = math.comb(num_files, num_sources)
        
        if self.max_mixtures is not None and max_combinations > self.max_mixtures:
            self.repeat_factor = 1
            self.total_mixtures = self.max_mixtures
        else:
            self.repeat_factor = max(1, math.ceil(min_mixtures / max_combinations))
            self.total_mixtures = max_combinations * self.repeat_factor
        
        print(f"Dataset will generate {self.total_mixtures} unique mixtures (repeat factor: {self.repeat_factor})")
        
        self.combinations = list(combinations(range(len(self.audio_files)), num_sources))
        
        if self.max_mixtures is not None and len(self.combinations) > self.max_mixtures:
            import random
            self.combinations = random.sample(self.combinations, self.max_mixtures)
        else:
            import random
            random.shuffle(self.combinations)

    def __len__(self):
        return len(self.combinations) * self.repeat_factor

    def _load_audio(self, path):
        """Load and standardize a single audio file with caching support"""
        if self.use_cache and path in self.audio_cache:
            return self.audio_cache[path].clone()
            
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, self.segment_samples)
            
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        return waveform

    def _normalize_audio(self, waveform):
        """Normalize audio based on selected method"""
        if self.normalize_method == 'unit_max':
            max_val = waveform.abs().max() + 1e-8
            return 0.9 * (waveform / max_val)
        elif self.normalize_method == 'rms':
            rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
            target_rms = 0.05  
            return waveform * (target_rms / rms)
        else:  
            return waveform

    def _apply_augmentation(self, waveform):
        """Apply random augmentation to the audio"""
        if not self.augment:
            return waveform
            
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            waveform = waveform * scale
        
        if random.random() > 0.5:
            waveform = -waveform
        
        if random.random() > 0.7:
            shift = random.randint(-100, 100)  
            if shift > 0:
                waveform = torch.nn.functional.pad(waveform[:, :-shift], (shift, 0))
            elif shift < 0:
                waveform = torch.nn.functional.pad(waveform[:, -shift:], (0, -shift))
            
        return waveform

    def __getitem__(self, idx):
        combination_idx = idx % len(self.combinations)
        repeat_idx = idx // len(self.combinations)
      
        source_indices = self.combinations[combination_idx]
        sources = []
        
        for src_idx in source_indices:
            waveform = self._load_audio(self.audio_files[src_idx])
            
            variation_seed = hash((src_idx, repeat_idx)) % 1000
            random.seed(variation_seed)
            
            if waveform.shape[1] >= self.segment_samples:
                energy = waveform.abs().mean(dim=0)
                threshold = energy.mean() * 0.1
                valid_positions = torch.where(energy > threshold)[0]
                
                if len(valid_positions) > self.segment_samples:
                    valid_starts = valid_positions[valid_positions < (waveform.shape[1] - self.segment_samples)]
                    if len(valid_starts) > 0:
                        start_idx = (variation_seed + repeat_idx) % len(valid_starts)
                        start = valid_starts[start_idx].item()
                    else:
                        start = random.randint(0, waveform.shape[1] - self.segment_samples)
                else:
                    start = random.randint(0, waveform.shape[1] - self.segment_samples)
                    
                waveform = waveform[:, start:start + self.segment_samples]
            else:
                padding = self.segment_samples - waveform.shape[1]
                left_pad = padding // 2
                right_pad = padding - left_pad
                waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
           random.seed()
            
            waveform = self._normalize_audio(waveform)
            
            waveform = self._apply_augmentation(waveform)
            
            sources.append(waveform.squeeze(0))
       
        sources = torch.stack(sources)
        
        if self.min_snr != self.max_snr:
            energy = torch.mean(sources**2, dim=1, keepdim=True)
          
            snr_db = torch.tensor([random.uniform(self.min_snr, self.max_snr) for _ in range(self.num_sources - 1)])
            snr_linear = 10 ** (snr_db / 20)
            
            scaling_factors = torch.ones(self.num_sources)
            
            for i in range(1, self.num_sources):
                # Calculate scaling to achieve target SNR
                target_energy = energy[0] / snr_linear[i-1]**2
                scaling_factors[i] = torch.sqrt(target_energy / (energy[i] + 1e-8))
                
            sources = sources * scaling_factors.view(-1, 1)
        
        mixture = sources.sum(dim=0)
      
        if mixture.abs().max() > 0.99:
            mixture = 0.99 * mixture / mixture.abs().max()
        
        if mixture.abs().max() < 0.01 or any(s.abs().max() < 0.01 for s in sources):
            # Generate synthetic noise mixture as fallback
            sources = torch.randn(self.num_sources, self.segment_samples) * 0.05
            mixture = sources.sum(dim=0)
            
        return mixture, sources

    @staticmethod
    def collate_fn(batch):
        """Batch processing with better handling of variable length audio"""
        mixtures, sources = zip(*batch)
      
        max_len = max(m.shape[0] for m in mixtures)
        def pad(x):
            if x.shape[0] < max_len:
                return torch.nn.functional.pad(x, (0, max_len - x.shape[0]))
            return x
        
        mixtures = torch.stack([pad(m) for m in mixtures])
        
        padded_sources = []
        for src_set in sources:
            padded_set = []
            for s in src_set:
                padded_set.append(pad(s))
            padded_sources.append(torch.stack(padded_set))
        
        sources = torch.stack(padded_sources)
        
        return mixtures, sources
