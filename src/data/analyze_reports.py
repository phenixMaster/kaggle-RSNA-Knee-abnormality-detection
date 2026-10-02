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
    config_file_path = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it-assistant/1/config.json"
    model_path = str(Path(config_file_path).parent)
    
    print(f"Attempting to load model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # Set padding token to eos_token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            dtype=torch.bfloat16,
            local_files_only=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if not os.path.exists(labels_csv):
        print(f"Error: {labels_csv} not found.")
        return
        
    df = pd.read_csv(labels_csv)
    target_cols = [col for col in df.columns if col not in ["StudyInstanceUID", "Report"]]
    
    # Prepare the list of target labels for the prompt
    labels_list_str = ", ".join(target_cols)
    
    results = []
    batch_size = 4 # Adjust based on VRAM
    
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
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=False
            )
        
        # Decode and parse
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for idx, response in enumerate(decoded_outputs):
            study_id = batch_df.iloc[idx]['StudyInstanceUID']
            study_scores = {"StudyInstanceUID": study_id}
            
            # Extract only the generated part after "Scores:"
            res_text = response.split("Scores:")[-1].strip()
            scores = [s.strip() for s in res_text.split(",")]
            
            for j, col in enumerate(target_cols):
                try:
                    if j < len(scores):
                        val = float(scores[j].split()[0]) # Handle "0.5" or "0.5 (certain)"
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

