import os
import pandas as pd
import torch
from pathlib import Path
import keras_nlp
from tqdm import tqdm

def main():
    # Configuration
    base_path = "/kaggle/input/competitions/rsna-knee-abnormality-detection" if os.path.exists("/kaggle/input") else "./data/raw"
    labels_csv = os.path.join(base_path, "train.csv")
    output_csv = "train_text_labels.csv"
    
    # Chemin du modèle Keras
    model_path = "/kaggle/input/models/keras/gemma4/keras/gemma4_2b/2"
    
    print(f"Loading Keras model from: {model_path}...")
    
    try:
        # Chargement via KerasNLP
        gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if not os.path.exists(labels_csv):
        print(f"Error: {labels_csv} not found.")
        return
        
    df = pd.read_csv(labels_csv)
    target_cols = [col for col in df.columns if col not in ["StudyInstanceUID", "Report"]]
    
    labels_list_str = ", ".join(target_cols)
    results = []
    
    batch_size = 4 
    print(f"Analyzing reports in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        
        prompts = []
        for _, row in batch_df.iterrows():
            prompt = (
                f"You are an expert musculoskeletal radiologist. Analyze the following MRI report "
                f"and provide a probability score (0.0 to 1.0) for each of these abnormalities: {labels_list_str}. "
                f"Format your response as a comma-separated list of numbers only, in the same order as the labels. "
                f"If not mentioned, use 0.0.\n\n"
                f"Report: {row['Report']}\n\n"
                f"Scores:"
            )
            prompts.append(prompt)
        
        # Génération via KerasNLP
        responses = gemma_lm.generate(prompts, max_length=128)
        
        for idx, response in enumerate(responses):
            study_id = batch_df.iloc[idx]['StudyInstanceUID']
            study_scores = {"StudyInstanceUID": study_id}
            
            res_text = response.split("Scores:")[-1].strip()
            scores = [s.strip() for s in res_text.split(",")]
            
            for j, col in enumerate(target_cols):
                try:
                    if j < len(scores):
                        val = float(scores[j].split()[0])
                        score = max(0.0, min(1.0, val))
                    else:
                        score = 0.0
                except:
                    score = 0.0
                study_scores[col] = score
                
            results.append(study_scores)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

