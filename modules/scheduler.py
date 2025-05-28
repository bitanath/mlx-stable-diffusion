import mlx.core as mx
import numpy as np

class EulerAncestralSampler:
    def __init__(self, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # These params are as in the original paper. I dont really know what these mean feel free to open an issue if anything seems off.
        self.betas = mx.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=mx.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        self.num_train_timesteps = num_training_steps
        self.timesteps = mx.array(np.arange(0, num_training_steps)[::-1].copy())
        
        self.sigmas = mx.sqrt((1 - self.alphas_cumprod) / mx.clip(self.alphas_cumprod, 1e-8, None))
        self.sigmas = mx.concatenate([mx.zeros(1), self.sigmas])

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = mx.array(timesteps)
        
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return max(prev_t, 0)  # NOTE: don't go below 0
    
    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: mx.array, model_output: mx.array, debug: bool = False):
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        if debug:
            print(f"Current timestep: {t}, Previous timestep: {prev_t}")
        
        sigma = self.sigmas[t].astype(latents.dtype)
        sigma_prev = self.sigmas[prev_t].astype(latents.dtype)
        
        if debug:
            print(f"Sigma: {sigma.item()}, Sigma_prev: {sigma_prev.item()}")
        
        while len(sigma.shape) < len(latents.shape):
            sigma = mx.expand_dims(sigma, -1)
            sigma_prev = mx.expand_dims(sigma_prev, -1)
        
        sigma2 = mx.clip(sigma.power(2), 1e-8, None)
        sigma_prev2 = mx.clip(sigma_prev.power(2), 1e-8, None)
        
        sigma_up_arg = mx.clip((sigma_prev2 * (sigma2 - sigma_prev2) / sigma2), 1e-8, None)
        sigma_up = mx.sqrt(sigma_up_arg)
        
        sigma_down_arg = mx.clip((sigma_prev2 - sigma_up.power(2)), 1e-8, None)
        sigma_down = mx.sqrt(sigma_down_arg)
        
        if debug:
            print(f"Sigma_up: {sigma_up.reshape(-1)[0].item()}, Sigma_down: {sigma_down.reshape(-1)[0].item()}")
        
        dt = sigma_down - sigma
        
        x_t_prev = (sigma2 + 1).sqrt() * latents + model_output * dt
        
        if mx.isnan(x_t_prev).any():
            print("Warning: NaN detected after first part of update")
            x_t_prev = mx.where(mx.isnan(x_t_prev), mx.zeros_like(x_t_prev), x_t_prev)
        
        if prev_t > 0:
            noise = mx.random.normal(shape=x_t_prev.shape, dtype=x_t_prev.dtype)
            x_t_prev = x_t_prev + noise * sigma_up
        
        scale_factor = mx.clip(1.0 / mx.sqrt(sigma_prev2 + 1), 1e-8, 1e8)
        x_t_prev = x_t_prev * scale_factor
        
        #TODO: Final NaN check to prevent nasty surprises
        if mx.isnan(x_t_prev).any():
            print("Warning: NaN detected in final result")
            x_t_prev = mx.where(mx.isnan(x_t_prev), mx.zeros_like(x_t_prev), x_t_prev)
        
        return x_t_prev
    
    def add_noise(
        self,
        original_samples: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        alphas_cumprod = self.alphas_cumprod.astype(original_samples.dtype)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.reshape(-1)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, -1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, -1)

        noise = mx.random.normal(shape=original_samples.shape, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    

class DDPMSampler:
    def __init__(self, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = mx.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=mx.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)
        self.one = mx.array(1.0)

        self.num_train_timesteps = num_training_steps
        self.timesteps = mx.array(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = mx.array(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> mx.array:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = mx.clip(variance, 1e-20, None)

        return variance
    
    def set_strength(self, strength=1):
        
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: mx.array, model_output: mx.array):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            noise = mx.random.normal(shape=model_output.shape, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        alphas_cumprod = self.alphas_cumprod.astype(original_samples.dtype)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.reshape(-1)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, -1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, -1)

        noise = mx.random.normal(shape=original_samples.shape, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
