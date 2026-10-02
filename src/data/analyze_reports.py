import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import os
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main():
    # Configuration
    base_path = "/kaggle/input/competitions/rsna-knee-abnormality-detection" if os.path.exists("/kaggle/input") else "./data/raw"
    labels_csv = os.path.join(base_path, "train.csv")
    output_csv = "train_text_labels.csv"
    
    # Chemin exact du fichier config.json
    config_file_path = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b-it/1/config.json"
    model_path = str(Path(config_file_path).parent)
    
    print(f"Attempting to load model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # On charge le CSV après avoir réussi à charger le modèle
    if not os.path.exists(labels_csv):
        print(f"Error: {labels_csv} not found.")
        return
        
    df = pd.read_csv(labels_csv)
    target_cols = [col for col in df.columns if col not in ["StudyInstanceUID", "Report"]]
    
    results = []

    print("Analyzing reports...")
    # Note: use a small sample first if you want to test
    # df = df.head(10) 

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        study_id = row['StudyInstanceUID']
        report = row['Report']
        
        study_scores = {"StudyInstanceUID": study_id}
        
        for col in target_cols:
            prompt = (
                f"You are an expert musculoskeletal radiologist. Analyze the following MRI report "
                f"and provide a probability score between 0.0 and 1.0 for the presence of {col}. "
                f"Return ONLY the numerical value.\n\n"
                f"Report: {report}\n\n"
                f"Probability:"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=10)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                score_text = response.split("Probability:")[-1].strip().split()[0]
                score = float(score_text)
                score = max(0.0, min(1.0, score))
            except:
                score = 0.5
            
            study_scores[col] = score
            
        results.append(study_scores)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()

    df = pd.read_csv(labels_csv)
    target_cols = [col for col in df.columns if col not in ["StudyInstanceUID", "Report"]]
    
    results = []

    print("Analyzing reports...")
    for idx, row in tqdm(df.iterrows(), total=len(df)).as_stqdm():
        study_id = row['StudyInstanceUID']
        report = row['Report']
        
        study_scores = {"StudyInstanceUID": study_id}
        
        for col in target_cols:
            # Prompt optimisé pour l'extraction de probabilité
            prompt = (
                f"You are an expert musculoskeletal radiologist. Analyze the following MRI report "
                f"and provide a probability score between 0.0 and 1.0 for the presence of {col}. "
                f"Return ONLY the numerical value.\n\n"
                f"Report: {report}\n\n"
                f"Probability:"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=10)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraction du nombre dans la réponse
            try:
                # On récupère la partie après le prompt
                score_text = response.split("Probability:")[-1].strip().split()[0]
                score = float(score_text)
                score = max(0.0, min(1.0, score)) # Clip entre 0 et 1
            except:
                score = 0.5 # Valeur neutre en cas d'échec
            
            study_scores[col] = score
            
        results.append(study_scores)

    # Sauvegarde
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
