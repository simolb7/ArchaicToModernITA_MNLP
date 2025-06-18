import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse

def main(args):

    model_path = "/leonardo_scratch/large/userexternal/mdimarco/hf_cache/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/892b3d7a7b1cf10c7a701c60881cd93df615734c"


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Questo mappa automaticamente su cuda:0
        trust_remote_code=True
    )

    df = pd.read_csv(args.input_path)

    translated_sentences = []

    for sentence in df["Sentence"]:
        messages = [
            {"role": "system", 
            "content": "You are an expert translator specializing in Medieval Italian texts (13th-15th century). Translate the following text from Archaic Italian to Modern Italian following these precise rules:  PRESERVE the original punctuation exactly as written, MAINTAIN the original sentence structure and word order, UPDATE only archaic vocabulary and grammatical forms to modern equivalents,DO NOT paraphrase or interpret - translate literally, KEEP the same level of formality and register. Respond with ONLY the translated text, no explanations or comments. Respond with ONLY the translated text and nothing else. Do not add explanations, introductions, or comments."
            },
        
            {"role": "user",
            "content": sentence},
        ]

        # Applica chat_template per creare l'input tokenizzato
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        # Genera output
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            
        # Decodifica l'output
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # Estrai la parte dopo <|assistant|>
        if "<|assistant|>" in decoded:
            translated = decoded.split("<|assistant|>")[-1].strip()
        else:
            translated = decoded.strip()

        # Rimuove eventuali note come "(Formal Register)"
        translated = translated.split("\n")[0].strip()

        # Rimuove anche tag eventuali come "(Modern Italian)"
        unwanted = ["(Modern Italian)", "(Formal Register)", "(Informal Register)"]
        for tag in unwanted:
            translated = translated.replace(tag, "").strip()
        
        translated_sentences.append(translated)

    df["ModernSentence_Zephyr"] = translated_sentences
    
    df.to_csv(args.output_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate tha sentences using Zephyr LLM")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output dataset.")
    args = parser.parse_args()
    main(args)