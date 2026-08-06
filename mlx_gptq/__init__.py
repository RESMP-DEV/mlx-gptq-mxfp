"""GPTQ-calibrated quantization on CUDA, packed into standard MLX (mlx-lm) format.

Two stages:
  Stage A (CUDA box):  python -m mlx_gptq.calibrate  -> calibration artifacts (q/scales/biases)
  Stage B (Mac):       python -m mlx_gptq.pack       -> standard mlx-lm model directory
"""

__version__ = "0.1.0"
