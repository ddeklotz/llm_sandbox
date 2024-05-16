from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "When I wake up in the morning, I"

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)

print(tokenizer.batch_decode(generated_ids)[0])
