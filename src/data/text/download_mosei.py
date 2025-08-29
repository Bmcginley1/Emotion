import os
import urllib.request
from mmsdk.mmdatasdk import mmdataset
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = "data/raw/mosei"
os.makedirs(DATA_ROOT, exist_ok=True)

# List of all MOSEI .csd URLs
CSD_URLS = {
    "words": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWords.csd",
    "phones": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedPhones.csd",
    "glove_vectors": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWordVectors.csd",
    "COVAREP": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd",
    "OpenFace_2": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd",
    "FACET 4.2": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualFacet42.csd",
    "All Labels": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd",
    #"glove_vectors_with_sp": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedGloveVectors_with_SP.csd"
}

# -----------------------------
# DOWNLOAD FUNCTION WITH PROGRESS BAR
# -----------------------------
class DownloadProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.update(total_size - self.pbar.n)
            self.pbar.close()

def download_csd_files(url_dict, dest_folder):
    for name, url in url_dict.items():
        filename = os.path.join(dest_folder, url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {name} from {url} ...")
            
            # Create progress bar instance
            progress_bar = DownloadProgressBar()
            
            # Download with progress bar
            urllib.request.urlretrieve(url, filename, reporthook=progress_bar)
            
            # Show completion info
            file_size = os.path.getsize(filename) / (1024**3)  # Convert to GB
            print(f"✓ Saved to {filename} ({file_size:.2f} GB)")
        else:
            file_size = os.path.getsize(filename) / (1024**3)
            print(f"✓ {filename} already exists ({file_size:.2f} GB), skipping download.")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Step 1/2: Downloading CMU-MOSEI .csd files...")
    download_csd_files(CSD_URLS, DATA_ROOT)

    print("\nStep 2/2: Initializing mmdataset...")
    # Build recipe using local file paths
    recipe = {key: os.path.join(DATA_ROOT, url.split('/')[-1]) for key, url in CSD_URLS.items()}

    dataset = mmdataset(recipe)
    print("\nMOSEI dataset successfully loaded! Available sequences:")
    print(dataset.keys())
    
    # Show total dataset size
    total_size = sum(os.path.getsize(os.path.join(DATA_ROOT, f)) 
                    for f in os.listdir(DATA_ROOT) if f.endswith('.csd'))
    print(f"\nTotal dataset size: {total_size / (1024**3):.2f} GB")