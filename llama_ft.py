import transformers
import torch
import pandas as pd
import argparse


def main(args):
    #define the model and the pipeline
    model = "./lora_llama3"  # change this if different

    pipeline = transformers.pipeline(
        "text-generation",
        model = model,
        model_kwargs = {"torch_dtype": torch.bfloat16},
        device_map = "auto",
    )

    translated_sentences = []

    df = pd.read_csv(args.input_path)
    
    #generete the message to send to the model, with a different number of sentences for in-contex learning
    message = generateMessage(args.n_shot)

    #for every sentences to the translation
    for sentece in df["Sentence"]:
        messages = message + [
        {
            "role": "user", 
            "content": sentece
        }
        ]   

        outputs = pipeline(
            messages,
            max_new_tokens = 256,
        )
        
        #clean the output
        trad = outputs[0]["generated_text"][-1]["content"]
        trad_clean = trad.removeprefix("Output: ")
        trad_clean = trad_clean.removeprefix("output: ")
        translated_sentences.append(trad_clean)
        
    df['ModernSentence_Llama_FT'] = translated_sentences

    #save the dataset with the translated sentences
    df.to_csv(args.output_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate tha sentences using Llama model and in-contex learining")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output dataset.")
    args = parser.parse_args()
    main(args)