import os
import librosa
import math
import json
import pdb

DATASET_PATH = 'virufy_data-main/clinical/segmented'
JSON_PATH = 'dataVirufy.json'
SAMPLE_RATE=22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # create dictionary
    data = {
        "mapping": [],
        "MFCCs": [],
        "labels": [],
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/") # Extracted_data/20200413 => ["Extracted_data", "20200413"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}". format(semantic_label))
            # process files for a spesific genre
            for f in filenames:
                
                #load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # extract the MFCCs 
                mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                
                data["MFCCs"].append(mfccs.T.tolist())

                data["labels"].append(i-1)

                
    
    with open(json_path, "w") as fp:

        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)