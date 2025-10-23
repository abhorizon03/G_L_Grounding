""" Script to generate sample images from the diffusion model.

In the generation of the images, the script is using a DDIM scheduler.
"""

import functools
import numpy as np
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from generative.networks.schedulers.ddim import DDIMPredictionType
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer


def set_timesteps_new(
        self, 
        num_inference_steps: int, 
        timestep_spacing: str = "leading",
        device: str | torch.device | None = None
) -> None:
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

    Args:
        num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
        device: target device to put the data.
    """
    if num_inference_steps > self.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
            f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps
    step_ratio = self.num_train_timesteps // num_inference_steps
    # creates integer timesteps by multiplying by ratio
    # casting to int to avoid issues when num_inference_step is power of 3
    # "leading" and "trailing" corresponds to annotation of Table 1. of https://arxiv.org/abs/2305.08891
    if timestep_spacing == "leading":
        timesteps = (np.arange(0, num_inference_steps)[::-1] * step_ratio).round().copy().astype(np.int64)
    elif timestep_spacing == "trailing":
        timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
        timesteps -= 1
    else:
        raise ValueError(
            f"{timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
        )
    self.timesteps = torch.from_numpy(timesteps).to(device)


def reversed_step_new(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the next timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.

        Returns:
            pred_prev_sample: Predicted previous sample
            pred_original_sample: Predicted original sample
        """
        # See Appendix F at https://arxiv.org/pdf/2105.05233.pdf, or Equation (6) in https://arxiv.org/pdf/2203.04306.pdf

        # Notation (<variable name> -> <name in paper>
        # - model_output -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_post_sample -> "x_t+1"

        # 1. get previous step value (=t+1)
        prev_timestep = timestep
        timestep = min(
            timestep - self.num_train_timesteps // self.num_inference_steps, self.num_train_timesteps - 1
        )

        # 2. compute alphas, betas at timestep t+1
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf

        if self.prediction_type == DDIMPredictionType.EPSILON:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.prediction_type == DDIMPredictionType.SAMPLE:
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.prediction_type == DDIMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

        # 4. Clip "predicted x_0"
        if self.clip_sample:
            #pred_original_sample = clamp_to_spatial_quantile(pred_original_sample)
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 6. compute x_t+1 without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_post_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return pred_post_sample, pred_original_sample


def get_modules(
        stage1_config_file_path, stage1_path, diffusion_config_file_path, diffusion_path, 
        device=torch.device("cuda"), num_timesteps=300
    ):
    
    config = OmegaConf.load(stage1_config_file_path)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(stage1_path))
    stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(diffusion_config_file_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(diffusion_path))
    diffusion.to(device)
    diffusion.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    # update set_timesteps method
    scheduler.set_timesteps = functools.partial(set_timesteps_new, scheduler)
    scheduler.set_timesteps(num_timesteps, "trailing", device)


    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
    
    return stage1, diffusion, scheduler, tokenizer, text_encoder

    
def get_prompt_embeds(prompt, tokenizer, text_encoder):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.squeeze(1))
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds