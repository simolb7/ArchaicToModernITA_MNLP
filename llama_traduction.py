import transformers
import torch
import pandas as pd
import argparse

def generateMessage(n: int):
    all_sentences = [
        {
            "role": "user",
            "content": "Input: Amor, ch'a nullo amato amar perdona, mi prese del costui piacer sì forte, che, come vedi, ancor non m'abbandona."
        },
        {
            "role": "assistant",
            "content": "Output: Amore, che non tollera che chi è amato non ami a sua volta, mi rapì della bellezza di questi [Paolo] in modo così potente, che, come vedi, ancora lo amo."
        },
        
        {
            "role": "user",
            "content": "Input: Considerate la vostra semenza: fatti non foste a viver come bruti, ma per seguir virtute e canoscenza"
        },
        {
            "role": "assistant",
            "content": "Output: Prendete coscienza della vostra condizione [di uomini]: non foste creati per vivere come selvaggi, ma per accrescere [le vostre] virtù e [il vostro] sapere."
        },
        
        {
            "role": "user",
            "content": "Input: Dette mi fuor di mia vita futura parole gravi, avvegna ch'io mi senta ben tetragono ai colpi di ventura."
        },
        {
            "role": "assistant",
            "content": "Output: A proposito della mia vita futura mi sono state rivolte parole dure e preoccupanti, sebbene io mi senta ben stabile davanti ai colpi della sorte."
        },
        
        {
            "role": "user",
            "content": "Input: A l'alta fantasia qui mancò possa; ma già volgeva il mio disio e 'l velle, sì come rota ch'igualmente è mossa, l'amor che move il sole e l'altre stelle"
        },
        {
            "role": "assistant",
            "content": "Output: All'immaginazione ora mancò la capacità, ma già il mio desiderio ed il volere erano soddisfatti, come una ruota che si muove di moto uniforme, dall'amor che muove il sole e le altre stelle."
        },
        
        {
            "role": "user",
            "content": "Input: Ahi serva Italia, di dolore ostello, nave sanza nocchiere in gran tempesta, non donna di provincie, ma bordello"
        },
        {
            "role": "assistant",
            "content": "Output: Ahi, Italia, schiava, albergo di dolore, nave senza guida in una tempesta, non donna rispettabile, ma prostituta!"
        },
        
        {
            "role": "user",
            "content": "Input: Sovra candido vel cinta d'uliva donna m'apparve, sotto verde manto vestita di color di fiamma viva."
        },
        {
            "role": "assistant",
            "content": "Output: Posta su un candido velo coronata d'ulivo mi apparve una donna, coperta da un manto verde con un vestito di color rosso vivo"
        },
        
        {
            "role": "user",
            "content": "Input: Era già l'ora che volge il disio ai navicanti e 'ntenerisce il core lo dì c' han detto ai dolci amici addio;"
        },
        {
            "role": "assistant",
            "content": "Output: Ormai si era fatta quell'ora che fa sì che, a coloro che navigano, in quel giorno che hanno detto addio ai cari amici, il desiderio si volga indietro e il cuore si intenerisca "
        },
    ]
    
    message = [
        {
            "role": "system", 
            "content": "You are an expert translator specializing in Medieval Italian texts (13th-15th century). Translate the following text from Archaic Italian to Modern Italian following these precise rules:  PRESERVE the original punctuation exactly as written, MAINTAIN the original sentence structure and word order, UPDATE only archaic vocabulary and grammatical forms to modern equivalents,DO NOT paraphrase or interpret - translate literally, KEEP the same level of formality and register. Respond with ONLY the translated text, no explanations or comments."
        },] + all_sentences[0:n*2]
    
    return message


def main(args):
    #define the model and the pipeline
    model = "/leonardo_scratch/large/userexternal/mdimarco/hf_cache/hub/models--meta-llama--LLama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"

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
        
    name_colomn = "ModernSentence_Llama_" + str(args.n_shot) + "Shot"
    df[name_colomn] = translated_sentences

    #save the dataset with the translated sentences
    df.to_csv(args.output_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate tha sentences using Llama model and in-contex learining")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output dataset.")
    parser.add_argument("--n_shot", type=int, required=True, help="Number of sentences add to the input messages for in-contex learning")
    args = parser.parse_args()
    main(args)