from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype = "float16",
    bnb_4bit_quant_type = "nf4",

)


model = AutoModelForCausalLM.from_pretrained("E:\\HumorGeneration\\mistral_finetuned\\checkpoint-45", device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("E:\\HumorGeneration\\mistral_finetuned\\checkpoint-45")

messages = [
    {"role": "user", "content": "You are a helpful joker that generates funny jokes. Please generate a funny joke that contains the following words: \"w1\" and \"w2\". Please generate only the joke!"},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])