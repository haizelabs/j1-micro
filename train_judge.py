import torch
from config import Config
from peft import LoraConfig
from rewards import spct_format_reward_func, spct_argmax_reward_func
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import GRPOConfig, GRPOTrainer
from utils import get_skywork_dataset

train_dataset = get_skywork_dataset(Config.TRAIN_INPUT_FILE)
test_dataset = get_skywork_dataset(Config.TEST_INPUT_FILE)

lora_config = LoraConfig(
    r=Config.LORA_RANK,
    lora_alpha=Config.LORA_ALPHA,
    lora_dropout=Config.LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    Config.MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_state_dict=True,
    attn_implementation="flash_attention_2",
)
model.enable_input_require_grads()
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

training_args = GRPOConfig(
    run_name=Config.RUN_NAME,
    output_dir=Config.OUTPUT_DIR,
    use_vllm=Config.USE_VLLM,
    vllm_device=Config.VLLM_DEVICE,
    vllm_gpu_memory_utilization=Config.VLLM_USAGE,
    num_generations=Config.GROUP_NUM,
    max_prompt_length=Config.MAX_PROMPT_LENGTH,
    max_completion_length=Config.MAX_SEQ_LENGTH - Config.MAX_PROMPT_LENGTH,
    vllm_max_model_len=Config.MAX_SEQ_LENGTH,
    per_device_train_batch_size=Config.BATCH_SIZE * Config.GROUP_NUM,
    gradient_accumulation_steps=Config.GRAD_ACCUM,
    num_train_epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    eval_steps=Config.EVAL_STEPS,
    eval_strategy="steps",
    gradient_checkpointing=True,
    beta=0.0005,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_steps=5,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=True,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_completions=True,
    overwrite_output_dir=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    peft_config=lora_config,
    reward_funcs=[spct_format_reward_func, spct_argmax_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
model.save_lora("spct-mini-lora")
