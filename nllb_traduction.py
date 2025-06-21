from transformers import pipeline
import torch
import pandas as pd
import argparse

def main(args):
    #define the model and the pipeline
    model_path = "/leonardo_scratch/large/userexternal/mdimarco/hf_cache/hub/models--facebook--nllb-200-distilled-1.3B/snapshots/7be3e24664b38ce1cac29b8aeed6911aa0cf0576"

    translator = pipeline(
        "translation",
        model=model_path,
        tokenizer=model_path,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    translated_sentences = []

    df = pd.read_csv(args.input_path)

    #for every sentences in the dataset do a translation
    for sentence in df["Sentence"]:

        translated = translator(sentence, src_lang="ita_Latn", tgt_lang="ita_Latn") 
        translated_sentences.append(translated[0]['translation_text'])

    df["ModernSentence_NLLB"] = translated_sentences

    #save the output dataset with the translated sentences
    df.to_csv(args.output_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate tha sentences using NLLB trasnformer-base translation machine")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output dataset.")
    args = parser.parse_args()
    main(args)