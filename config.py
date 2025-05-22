class Config:
    # VLLM settings
    USE_VLLM = True
    VLLM_DEVICE = "cuda:0"
    VLLM_USAGE = 0.2

    # Training settings
    EPOCHS = 10
    GROUP_NUM = 2
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 1e-4
    EVAL_STRATEGY = "no"
    EVAL_STEPS = 1000

    # Model settings
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    MAX_SEQ_LENGTH = 4096
    MAX_PROMPT_LENGTH = 2048
    LORA_RANK = 16
    LORA_ALPHA = LORA_RANK * 2
    LORA_DROPOUT = 0.1

    # Input settings
    TRAIN_INPUT_FILE = "train_df.csv"
    TEST_INPUT_FILE = "test_df.csv"
    
    # Output settings
    OUTPUT_DIR = "j1-micro-lora"
    RUN_NAME = OUTPUT_DIR
    WANDB_PROJECT = "j1-micro-judge"

    # Column names from Skywork v2.0 dataset
    COLUMN_INPUT = "sky_input"
    COLUMN_CHOSEN_POSITION = "chosen_positions"
    COLUMN_CHOSEN = "sky_chosen"
    COLUMN_REJECTED = "sky_rejected"
    COLUMN_PROMPT = "prompt"
    COLUMN_SOURCE = "source"
    COLUMN_CHOSEN_ORIG = "chosen"
    COLUMN_REJECTED_ORIG = "rejected"

    # Chosen position
    CHOSEN_POSITION_A = "a"
    CHOSEN_POSITION_B = "b"
