from modal import Stub, method, Image, Mount
from modal_config import configure_base_image, MODEL_VOLUME, GPU_CONFIG, MODEL_URLS, ENV_VARS
import os
import torch
from pathlib import Path
import huggingface_hub
from omegaconf import OmegaConf

# Create stub
stub = Stub("index-anisora")

# Configure image
image = configure_base_image()

@stub.cls(
    image=image,
    gpu=GPU_CONFIG,
    network_file_systems={"/model_cache": MODEL_VOLUME},
    secret=Secret.from_name("huggingface-secret"),
    mounts=[Mount.from_local_dir("configs", remote_path="/root/configs")]
)
class AnisoraModel:
    def __enter__(self):
        # Set environment variables
        for key, value in ENV_VARS.items():
            os.environ[key] = value

        # Create necessary directories
        self.model_dir = Path("/model_cache")
        self.pretrained_dir = self.model_dir / "pretrained_models"
        self.ckpt_dir = self.model_dir / "ckpt"

        for dir in [self.pretrained_dir, self.ckpt_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # Download model weights if not present
        self._download_model_weights()

        # Initialize model
        self._initialize_model()

    def _download_model_weights(self):
        """Download required model weights from HuggingFace"""
        # Use HuggingFace token from secrets
        huggingface_token = os.environ["HUGGINGFACE_TOKEN"]

        # Download text encoder and VAE
        if not (self.pretrained_dir / "text_encoder").exists():
            huggingface_hub.snapshot_download(
                "IndexTeam/Index-anisora",
                repo_type="model",
                token=huggingface_token,
                allow_patterns="CogVideoX_VAE_T5/*",
                local_dir=self.pretrained_dir
            )

        # Download 5B model weights
        if not (self.ckpt_dir / "model.safetensors").exists():
            huggingface_hub.snapshot_download(
                "IndexTeam/Index-anisora",
                repo_type="model",
                token=huggingface_token,
                allow_patterns="5B/*",
                local_dir=self.ckpt_dir
            )

    def _initialize_model(self):
        """Initialize the Anisora model"""
        from fastercache.models.cogvideox.diffusion_video import SATVideoDiffusionEngine

        # Load config
        config_path = "configs/cogvideox/cogvideox_5b_720_169_2.yaml"
        self.config = OmegaConf.load(config_path)

        # Initialize model
        self.model = SATVideoDiffusionEngine(self.config)
        self.model.eval()
        self.model.to("cuda")

    @method()
    def generate(self, prompt: str, num_frames: int = 49):
        """Generate video from text prompt"""
        try:
            # Prepare input batch
            batch = {
                "prompt": prompt,
                "num_frames": num_frames
            }

            # Generate video
            with torch.no_grad():
                output = self.model.log_video(
                    batch,
                    N=1,
                    only_log_video_latents=False
                )

            return {
                "video": output["samples"].cpu().numpy(),
                "status": "success"
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

# Example usage
@stub.local_entrypoint()
def main():
    model = AnisoraModel()
    result = model.generate(
        prompt="A beautiful anime girl with long blue hair"
    )
    print(f"Generation status: {result['status']}")