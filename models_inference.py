from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import re

models_path = {'mistral':'E:\\HumorGeneration\\mistral_finetuned\\checkpoint-45',
                'qwen0.5b':'E:\HumorGeneration\qwen0.5b\checkpoint-45', 
                'qwen4b': 'E:\HumorGeneration\qwen4b\checkpoint-45'}

test_data = pd.read_csv('task-a-en-in.tsv', sep='\t').drop(columns=['headline'])

def compute(row, model, tokenizer):
    messages = [
    {"role": "user", "content": f"You are a helpful joker that generates funny jokes. Please generate a funny joke that contains the following words: \"{row['word1']}\" and \"{row['word2']}\". Please output only the joke!"},]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    '''pattern = r"(\[\/INST\])(.+)(<\/s>)"
    print('----------------------------------------------')
    print(decoded[0])
    match = re.search(pattern, decoded[0], re.S)

    if match:
        return match.group(2).replace("\n", " ").strip()
    else:
        return 'null'''
    return decoded[0]

for model_item in models_path.items():

    device = "cuda"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype = "float16",
        bnb_4bit_quant_type = "nf4")

    model = AutoModelForCausalLM.from_pretrained(model_item[1], device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_item[1])

    test_data[model_item[0]] = test_data.apply(compute, axis=1, args=(model, tokenizer))
test_data.drop(columns=['word1', 'word2'])
test_data.to_csv("comparison.tsv", sep="\t", index=False)