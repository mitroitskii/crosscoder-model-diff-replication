from utils import *
from transformer_lens import ActivationCache
import tqdm

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"] # ex. 4096 * 128 = 512,256
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1) # ex. 512,256 // 1023 = 500
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1) # ex. 500 * 1023 = 511,500
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model), # ex. (511500, 2, 2304)
            dtype=DTYPES[cfg["enc_dtype"]],
            requires_grad=False,
        ).to(cfg["device"])  # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
            [
                estimated_norm_scaling_factor_A,
                estimated_norm_scaling_factor_B,
            ],
            device=cfg["device"],
            dtype=DTYPES[cfg["enc_dtype"]],
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        self.buffer_pointer = 0
        print("Refreshing the buffer!")
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.buffer_batches  # 16
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            batch_pointer = 0 # a pointer to the current batch in the all_tokens array
            for _ in tqdm.trange(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[
                    self.token_pointer + batch_pointer: 
                    self.token_pointer + min(batch_pointer + self.cfg["model_batch_size"], num_batches)]

                _, cache_A = self.model_A.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_A: ActivationCache

                _, cache_B = self.model_B.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_B: ActivationCache

                acts = torch.stack(
                    [cache_A[self.cfg["hook_point"]], cache_B[self.cfg["hook_point"]]], dim=0)
                acts = acts[:, :, 1:, :]  # Drop BOS
                # [2, batch, seq_len-1, d_model]
                assert acts.shape == (
                    2, tokens.shape[0], tokens.shape[1]-1, self.model_A.cfg.d_model) # tokens.shape[0] = model_batch_size
                # ex [2, 4, 1023, 2304]
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )  # ex [4092, 2, 2304]

                self.buffer[self.buffer_pointer: self.buffer_pointer +
                            acts.shape[0]] = acts
                self.buffer_pointer += acts.shape[0]
                batch_pointer = min(
                    batch_pointer + self.cfg["model_batch_size"], num_batches)

        self.token_pointer += num_batches
        self.buffer_pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.buffer_pointer: self.buffer_pointer +
                          self.cfg["batch_size"]].float()
        # out: [batch_size, 2, d_model]
        self.buffer_pointer += self.cfg["batch_size"]
        if self.buffer_pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
