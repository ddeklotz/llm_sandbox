import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def format_record_chatml(q, a):
    template="<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    return template.format(
        q=q,
        a=a,
    )

# Define a function to read the CSV file and convert entries into ChatML records
def csv_to_chatml(csv_file):
    chatml_records = []
    with open(csv_file, 'r', newline='', encoding='utf-8', errors='ignore') as file:
        csv_reader = csv.reader(file)
        # Skip the header row if it exists
        next(csv_reader, None)
        for row in csv_reader:
            # Extract values from each row
            date, word, clue = row
            chatml_records.append(format_record_chatml(word, clue))
    return chatml_records

def load_nythints():
    csv_file = ".\\nytcrosswords_shorter.csv"

    # Convert CSV entries into ChatML records
    chatml_records = csv_to_chatml(csv_file)
    return chatml_records

def main():    
    # Path to the CSV file
    csv_file = ".\\nytcrosswords.csv"

    # Convert CSV entries into ChatML records
    chatml_records = csv_to_chatml(csv_file)

    # Printing the ChatML records for verification
    for record in chatml_records:
        print(record)


def finetune():
    
    modelpath="mistralai/Mistral-7B-v0.1"

    print("loading base model...")

    # Load 4-bit quantized model
    model = AutoModelForCausalLM.from_pretrained(
        modelpath,    
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16,
    )

    # Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
    tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)   

    # Add tokens <|im_start|> and <|im_end|>, latter is special eos token 
    tokenizer.pad_token = "</s>"
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id

    # Add LoRA adapters to model
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=64, 
        lora_alpha=16, 
        target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
        lora_dropout=0.1, 
        bias="none", 
        modules_to_save = ["lm_head", "embed_tokens"],        # needed because we added new tokens to tokenizer/model
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False

    def tokenize(element):
        return tokenizer(
            element["text"],
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )

    print("loading nyt crossword dataset...")

    dataset = load_nythints()

    print ("tokenizing dataset...")

    dataset_tokenized = dataset.map(
        tokenize, 
        batched=True, 
        num_proc=os.cpu_count(),    # multithreaded
        remove_columns=["text"]     # don't need the strings anymore, we have tokens from here on
    )

    # collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
    def collate(elements):
        tokenlist=[e["input_ids"] for e in elements]
        tokens_maxlen=max([len(t) for t in tokenlist])  # length of longest input

        input_ids,labels,attention_masks = [],[],[]
        for tokens in tokenlist:
            # how many pad tokens to add for this sample
            pad_len=tokens_maxlen-len(tokens)

            # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
            input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
            labels.append( tokens + [-100]*pad_len )    
            attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

        batch={
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks)
        }
        return batch
    
    bs=8        # batch size
    ga_steps=1  # gradient acc. steps
    epochs=5
    steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        evaluation_strategy="steps",
        logging_steps=1,
        eval_steps=steps_per_epoch,  # eval and save once per epoch   
        save_steps=steps_per_epoch,
        gradient_accumulation_steps=ga_steps,
        num_train_epochs=epochs,
        lr_scheduler_type="constant",
        optim="paged_adamw_32bit",
        learning_rate=0.0002,
        group_by_length=True,
        fp16=True,
        ddp_find_unused_parameters=False,    # needed for training with accelerate
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
        args=args,
    )

    print("training...")

    trainer.train()

    print("merging with base model...")

    base_path="mistralai/Mistral-7B-v0.1"    # input: base model
    adapter_path="out/checkpoint-606"     # input: adapters
    save_to="models/Mistral-7B-finetuned"    # out: merged model ready for inference

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Add/set tokens (same 5 lines of code we used before training)
    tokenizer.pad_token = "</s>"
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id

    # Load LoRA adapter and merge
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(save_to)

finetune()
