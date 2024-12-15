from utils import *
from trainer import Trainer

DEVICE = "cuda:1"
ENCODINGS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DTYPE = "bf16"

def get_default_cfg(base_model=None):
    """Get default configuration, optionally using model parameters"""
    
    # NOTE just reducing precision cuts on tens of GB of memory

    cfg = {
        "seed": 49, # random seed
        "batch_size": 4096, # base number of hidden layer activations crosscoder is trained on per batch
        "buffer_mult": 128, # multiplier for buffer size
        "lr": 5e-5, # learning rate
        "num_tokens": 400_000_000,  # number of tokens in the dataset NOTE: look up the number of tokens in your dataset and put it here
        "l1_coeff": 2,  # l1 regularization coefficient in the loss function
        "beta1": 0.9,  # controls the decay rate of the first moment estimate (mean) of the gradients
        "beta2": 0.999,  # ontrols the decay rate of the second moment estimate (uncentered variance) of the gradients
        "dict_size": 2**14,  # size of the crosscoder's hidden layer
        "seq_len": 1024,  # sequence length of the given dataset NOTE: look up the seq len your data has and put it here
        "enc_dtype": DTYPE, # NOTE: this must match the dtype of models diffed 
        "model_name": "gemma-2-2b",
        "site": "resid_pre",
        "device": DEVICE, 
        "model_batch_size": 4, # number of dataset batches loaded into memory at once when forming the buffer
        "log_every": 100,
        "save_every": 30000,
        "dec_init_norm": 0.08,
        "hook_point": "blocks.14.hook_resid_pre",
        "wandb_project": "crosscoders-test-run", # NOTE replace with your project name
        "wandb_entity": "tentative",  # NOTE replace with your wandb username
    }

    # Add model-dependent config if model is provided
    if base_model is not None:
        cfg["d_in"] = base_model.cfg.d_model

    return cfg

def main():

    base_model = HookedTransformer.from_pretrained_no_processing( # use from_pretrained_no_processing (from_pretrained) for models with lower (higher) precision 
        "gemma-2-2b",
        device=DEVICE,
        torch_dtype=ENCODINGS[DTYPE]
    )

    chat_model = HookedTransformer.from_pretrained_no_processing( # use from_pretrained_no_processing (from_pretrained) for models with lower (higher) precision 
        "gemma-2-2b-it",
        device=DEVICE,
        torch_dtype=ENCODINGS[DTYPE]
    )

    all_tokens = load_pile_lmsys_mixed_tokens()

    cfg = get_default_cfg(base_model)

    trainer = Trainer(cfg, base_model, chat_model, all_tokens)
    trainer.train()


if __name__ == "__main__":
    main()