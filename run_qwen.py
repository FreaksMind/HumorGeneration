from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import evaluate
#model_name = "Qwen/Qwen3-4B-Instruct-2507"
model_name = "E:\HumorGeneration\qwen0.5b\checkpoint-45"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype = "float16",
    bnb_4bit_quant_type = "nf4",

)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_config,
)

prompt = "You are a helpful joke generator that creates funny jokes. Please generate a funny joke that contains the following words: \"moon\" and \"computer\". Please generate only the joke!"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)