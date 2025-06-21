import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse

def main(args):

    #define the model
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

    #for every sentences in the dataset we create a messages to do the translation
    for sentence in df["Sentence"]:
        messages = [
            {"role": "system", 
            "content": "You are an expert translator specializing in Medieval Italian texts (13th-15th century). Translate the following text from Archaic Italian to Modern Italian following these precise rules:  PRESERVE the original punctuation exactly as written, MAINTAIN the original sentence structure and word order, UPDATE only archaic vocabulary and grammatical forms to modern equivalents,DO NOT paraphrase or interpret - translate literally, KEEP the same level of formality and register. Respond with ONLY the translated text, no explanations or comments. Respond with ONLY the translated text and nothing else. Do not add explanations, introductions, or comments."
            },
        
            {"role": "user",
            "content": sentence},
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        #generete de output
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        #eclean the output
        if "<|assistant|>" in decoded:
            translated = decoded.split("<|assistant|>")[-1].strip()
        else:
            translated = decoded.strip()
        translated = translated.split("\n")[0].strip()
        unwanted = ["(Modern Italian)", "(Formal Register)", "(Informal Register)"]
        for tag in unwanted:
            translated = translated.replace(tag, "").strip()
        
        #add the output to the list of translated sentences
        translated_sentences.append(translated)

    df["ModernSentence_Zephyr"] = translated_sentences
    
    #save the output dataset with the translated sentences
    df.to_csv(args.output_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate tha sentences using Zephyr LLM")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output dataset.")
    args = parser.parse_args()
    main(args)