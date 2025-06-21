from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import pandas as pd
import re
import argparse
from scipy.stats import spearmanr

def judging_prometheus(path: str, translation: str):
  transformers.set_seed(42)
  
  #define the model and the prompt
  device = "cuda"

  model_id = "/leonardo_scratch/large/userexternal/mdimarco/hf_cache/hub/models--Unbabel--M-Prometheus-7B/snapshots/030fb74806e4228c466a98706a297d43b31ce5df"

  model = AutoModelForCausalLM.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

  ABSOLUTE_PROMPT = """###Task Description:
  An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
  1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
  2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
  3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
  4. Please do not generate any other opening, closing, and explanations.

  ###The instruction to evaluate:
  {instruction}

  ###Response to evaluate:
  {response}

  ###Score Rubrics:
  {rubric}

  ###Feedback: """


  rubric_data = {
    "criteria":"Is the model the converts old and archaic italian to Modern Italian",
    "score1_description":"Fails to convey the meaning: gibberish or irrelevant output",
    "score2_description":"Significant loss of meaning: Major errors in translation or grammar, Misleading or confusing phrasing",
    "score3_description":"Some meaning preserved: Modern grammar/vocabulary partly correct, One or more semantic inaccuracies,  Unnatural sentence structure",
    "score4_description":"Core meaning preserved: Mostly accurate and fluent modernization with minor errors, May sound a bit unnatural but still comprehensible",
    "score5_description":"Preserves all core ideas and intent: The modernized version is fluent, fully faithful to the original meaning, and idiomatically natural in modern Italian"
  }

  scores = []

  df = pd.read_csv(path)
  
  #choose on with translation do the scorining
  if translation == "NLLB":
    colomn_name = "ModernSentence_" + translation
  elif translation == "Zephyr":
    colomn_name = "ModernSentence_" + translation
  elif translation == "FT":
    colomn_name = "ModernSentence_Llama_" + translation
  else:
    colomn_name = "ModernSentence_Llama_" + translation + "Shot"

  #judge the first twenty trunslation
  for idx, row in df.head(20).iterrows():

    user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
        instruction=row["Sentence"],
        response=row[colomn_name],
        rubric=rubric_data
    )
    
    messages = [
        {"role": "user", "content": user_content},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)

    #clean the output to get only the score
    output = decoded[0]
    matches = re.findall(r'\b([1-5])\b', output)
    score = int(matches[-1]) if matches else None
    scores.append(score)

  return scores

def correlation(llm_score: list[int], translation: str):
  #calculate the spearman correlation using human and prometheus score
  if translation == "Zephyr":
    zephyr = [2, 2, 1, 3, 3, 1, 2, 4, 3, 2, 2, 2, 1, 1, 1, 1, 3, 3, 2, 2]
    rho, p_value = spearmanr(llm_score, zephyr)
  elif translation == "0":
    Llama_0Shot = [1, 1, 1, 3, 1, 1, 2, 3, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2]
    rho, p_value = spearmanr(llm_score, Llama_0Shot)
  elif translation == "1":
    Llama_1Shot = [3, 2, 1, 2, 3, 3, 2, 4, 3, 4, 1, 1, 2, 2, 4, 2, 1, 2, 2, 3]
    rho, p_value = spearmanr(llm_score, Llama_1Shot)
  elif translation == "3":
    Llama_3Shot = [2, 3, 5, 2, 3, 4, 2, 2, 4, 5, 3, 5, 4, 1, 4, 3, 1, 1, 3, 3]
    rho, p_value = spearmanr(llm_score, Llama_3Shot)
  elif translation == "5":
    Llama_5Shot = [4, 4, 2, 3, 3, 4, 4, 3, 3, 1, 4, 3, 5, 2, 3, 3, 2, 2, 4, 3]
    rho, p_value = spearmanr(llm_score, Llama_5Shot)
  elif translation == "7":
    Llama_7Shot = [3, 1, 2, 3, 3, 4, 1, 1, 4, 3, 5, 3, 5, 3, 4, 3, 2, 3, 3, 4]
    rho, p_value = spearmanr(llm_score, Llama_7Shot)
  elif translation == "NLLB":
    nllb = [2, 1, 2, 2, 2, 2, 1, 3, 1, 1, 3, 4, 3, 2, 2, 3, 5, 1, 2, 2]
    rho, p_value = spearmanr(llm_score, nllb)
  elif translation == "FT":
    Llama_FT = [1, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 3, 2, 2, 2, 3, 1, 1, 2]
    rho, p_value = spearmanr(llm_score, Llama_FT)
  return rho, p_value
  
def main(args):
  llm_score = judging_prometheus(args.input_path, args.translation)
  rho, p_val = correlation(llm_score, args.translation)
  print(llm_score)
  print(rho)
  print(p_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Prometheus as an LLM-judge and compute the correlation between human and LLM metrics.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the trasnlated dataset.")
    parser.add_argument("--translation", type=str, required=True, help="Value to idicate which translation evaluate.")
    args = parser.parse_args()
    main(args)

