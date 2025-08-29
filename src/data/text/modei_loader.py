import os
from mmsdk.mmdatasdk import mmdataset

DATA_ROOT = "Emotion/data/raw/mosei"

CSD_FILES = {
    "words": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedWords.csd"),
    "phones": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedPhones.csd"),
    "glove_vectors": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedWordVectors.csd"),
    #"COVAREP": os.path.join(DATA_ROOT, "CMU_MOSEI_COVAREP.csd"),
    #"OpenFace_2": os.path.join(DATA_ROOT, "CMU_MOSEI_VisualOpenFace2.csd"),
    #"FACET 4.2": os.path.join(DATA_ROOT, "CMU_MOSEI_VisualFacet42.csd"),
    "All Labels": os.path.join(DATA_ROOT, "CMU_MOSEI_Labels.csd"),
    #"glove_vectors_with_sp": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedGloveVectors_with_SP.csd")
}

# Initialize dataset once
dataset = mmdataset(CSD_FILES)