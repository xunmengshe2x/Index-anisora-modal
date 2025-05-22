from modal import Image, Secret, NetworkFileSystem

# Create network file system for persistent storage
MODEL_VOLUME = NetworkFileSystem.persisted("anisora-model-cache")

# Base image configuration
def configure_base_image():
    return (
        Image.debian_slim(python_version="3.10")
        .apt_install([
            "git",
            "git-lfs",  # Required for model downloads
            "ffmpeg",    # Required for video processing
            "libgl1-mesa-glx", # OpenCV dependency
            "libglib2.0-0"
        ])
        .pip_install([
            # Core dependencies from anisoraV1_infer/requirements.txt
            "torch==2.0.1",
            "torchvision==0.15.2",
            "numpy>=1.24.3",
            "omegaconf>=2.3.0",
            "einops>=0.6.1",
            "pytorch-lightning>=2.0.2",
            "transformers>=4.30.2",
            "safetensors>=0.3.1",
            "accelerate>=0.20.3",
            "deepspeed>=0.9.5",
            "diffusers>=0.18.3",
            "opencv-python>=4.7.0.72",
            "pillow>=9.5.0",
            "tqdm>=4.65.0",
            "huggingface_hub>=0.15.1"
        ])
        .run_commands(
            "git lfs install"
        )
    )

# GPU configuration
GPU_CONFIG = "A100"  # Based on repository performance notes

# Model URLs extracted from repository
MODEL_URLS = {
    "text_encoder_vae": "https://huggingface.co/IndexTeam/Index-anisora/tree/main/CogVideoX_VAE_T5",
    "model_5b": "https://huggingface.co/IndexTeam/Index-anisora/tree/main/5B",
    # Alternative ModelScope URLs
    "modelscope_base": "https://modelscope.cn/models/bilibili-index/Index-anisora/files"
}

# Environment variables
ENV_VARS = {
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "6000",
    "LOCAL_RANK": "0",
    "RANK": "0",
    "WORLD_SIZE": "1"
}