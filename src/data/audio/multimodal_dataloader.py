import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from mmsdk.mmdatasdk import mmdataset
from sklearn.preprocessing import StandardScaler
import warnings

class MultimodalMOSEIDataset(Dataset):
    def __init__(self, text_features, audio_features, labels):
        self.text_features = text_features
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        return {
            'text': self.text_features[idx],
            'audio': self.audio_features[idx],
            'labels': self.labels[idx]
        }

def downsample_audio(audio_seq, target_length):
    """Downsample audio sequence to match target length"""
    current_length = audio_seq.shape[0]
    if current_length <= target_length:
        # Pad if audio is shorter than target
        padding = np.zeros((target_length - current_length, audio_seq.shape[1]))
        return np.vstack([audio_seq, padding])
    else:
        # Downsample using uniform sampling
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return audio_seq[indices]

def clean_audio_features(audio_features):
    """Clean audio features: handle inf/nan values and outliers"""
    # Replace inf/-inf with reasonable values
    audio_features = np.nan_to_num(
        audio_features, 
        nan=0.0, 
        posinf=10.0,  # Cap positive infinity
        neginf=-10.0  # Cap negative infinity
    )
    
    # Clip extreme outliers (beyond 3 standard deviations)
    mean = np.mean(audio_features)
    std = np.std(audio_features)
    audio_features = np.clip(audio_features, mean - 3*std, mean + 3*std)
    
    return audio_features

def save_processed_data(text_padded, audio_padded, labels_padded, audio_scaler, 
                       filepath="data/processed/multimodal_data.pt"):
    """Save processed data to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'text': text_padded,
        'audio': audio_padded, 
        'labels': labels_padded,
        'audio_scaler': audio_scaler
    }, filepath)
    print(f"Saved processed data to {filepath}")

def load_processed_data(filepath="data/processed/multimodal_data.pt"):
    """Load processed data from disk"""
    if os.path.exists(filepath):
        print(f"Loading cached data from {filepath}")
        return torch.load(filepath, weights_only=False)
    return None

def get_multimodal_dataloader(batch_size=16, shuffle=True, max_audio_length=500, use_cache=True):
    """
    Create dataloader for text + audio + labels
    
    Args:
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        max_audio_length: Maximum length for audio sequences (downsampled)
        use_cache: Whether to use cached processed data
    """
    
    # Try to load cached data first
    if use_cache:
        cached_data = load_processed_data()
        if cached_data is not None:
            text_padded = cached_data['text']
            audio_padded = cached_data['audio']
            labels_padded = cached_data['labels']
            audio_scaler = cached_data['audio_scaler']
            
            print(f"Loaded cached data shapes:")
            print(f"  Text: {text_padded.shape}")
            print(f"  Audio: {audio_padded.shape}")
            print(f"  Labels: {labels_padded.shape}")
            
            # Create dataset and dataloader from cached data
            dataset = MultimodalMOSEIDataset(text_padded, audio_padded, labels_padded)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return dataloader, audio_scaler
    
    # Process data from scratch if cache not found or not using cache
    print("Processing data from scratch...")
    DATA_ROOT = "data/raw/mosei"
    CSD_FILES = {
        "glove_vectors": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedWordVectors.csd"),
        "COVAREP": os.path.join(DATA_ROOT, "CMU_MOSEI_COVAREP.csd"),
        "All Labels": os.path.join(DATA_ROOT, "CMU_MOSEI_Labels.csd")
    }
    
    print("Loading multimodal MOSEI dataset...")
    dataset_obj = mmdataset(CSD_FILES)
    
    text_seq = dataset_obj["glove_vectors"]
    audio_seq = dataset_obj["COVAREP"]
    labels_seq = dataset_obj["All Labels"]
    
    # Find common utterances
    common_utt_ids = set(text_seq.data.keys()).intersection(
        set(audio_seq.data.keys())).intersection(
        set(labels_seq.data.keys()))
    
    print(f"Processing {len(common_utt_ids)} common utterances...")
    
    all_text = []
    all_audio = []
    all_labels = []
    
    # Collect all audio features for normalization
    all_audio_raw = []
    
    # Progress tracking
    from tqdm import tqdm
    
    for utt_id in tqdm(common_utt_ids, desc="Processing utterances"):
        try:
            # Extract features
            text_feat = text_seq.data[utt_id]["features"][:].astype('float32')
            audio_feat = audio_seq.data[utt_id]["features"][:].astype('float32')
            label_feat = labels_seq.data[utt_id]["features"][:].astype('float32')
            
            # Clean audio features
            audio_feat = clean_audio_features(audio_feat)
            
            # Ensure minimum sequence lengths
            if text_feat.shape[0] < 5 or audio_feat.shape[0] < 10:
                continue
                
            # Downsample audio to manageable length
            target_audio_length = min(max_audio_length, text_feat.shape[0] * 2)
            audio_feat_downsampled = downsample_audio(audio_feat, target_audio_length)
            
            # Handle label alignment - use the same approach as your original code
            min_len = min(text_feat.shape[0], label_feat.shape[0])
            text_feat = text_feat[:min_len]
            label_feat = label_feat[:min_len]
            
            all_text.append(torch.tensor(text_feat))
            all_audio.append(torch.tensor(audio_feat_downsampled))
            all_labels.append(torch.tensor(label_feat))
            
            # Store for normalization
            all_audio_raw.append(audio_feat_downsampled)
            
        except Exception as e:
            warnings.warn(f"Skipping utterance {utt_id}: {e}")
            continue
    
    print(f"Successfully processed {len(all_text)} utterances")
    
    # Normalize audio features
    print("Normalizing audio features...")
    all_audio_concat = np.vstack(all_audio_raw)
    audio_scaler = StandardScaler()
    audio_scaler.fit(all_audio_concat)
    
    # Apply normalization to all audio sequences
    normalized_audio = []
    for audio_seq in all_audio_raw:
        normalized_seq = audio_scaler.transform(audio_seq)
        normalized_audio.append(torch.tensor(normalized_seq.astype('float32')))
    
    # Pad sequences
    text_padded = pad_sequence(all_text, batch_first=True)
    audio_padded = pad_sequence(normalized_audio, batch_first=True)
    labels_padded = pad_sequence(all_labels, batch_first=True)
    
    print(f"Final shapes:")
    print(f"  Text: {text_padded.shape}")
    print(f"  Audio: {audio_padded.shape}")
    print(f"  Labels: {labels_padded.shape}")
    
    # Create dataset and dataloader
    dataset = MultimodalMOSEIDataset(text_padded, audio_padded, labels_padded)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Save processed data for future use
    if use_cache:
        save_processed_data(text_padded, audio_padded, labels_padded, audio_scaler)
    
    return dataloader, audio_scaler

if __name__ == "__main__":
    # Test the dataloader
    dataloader, scaler = get_multimodal_dataloader(batch_size=4, max_audio_length=300)
    
    # Test a batch
    for batch in dataloader:
        print("Batch shapes:")
        print(f"  Text: {batch['text'].shape}")
        print(f"  Audio: {batch['audio'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        
        print("Feature ranges:")
        print(f"  Text: [{batch['text'].min():.3f}, {batch['text'].max():.3f}]")
        print(f"  Audio: [{batch['audio'].min():.3f}, {batch['audio'].max():.3f}]")
        print(f"  Labels: [{batch['labels'].min():.3f}, {batch['labels'].max():.3f}]")
        break