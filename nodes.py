"""
Universal Smart VAE Decode - Claude Edition
Production-ready with best practices from GPT, Gemini, and Claude.

Author: Claude (Anthropic) - AI Ensemble collaboration
Version: 1.0.0
License: MIT
GitHub: https://github.com/uczensokratesa/ComfyUI-Claude-VAE

Features:
- Sliding window with temporal overlap (GPT's brilliance)
- Defensive parameter validation (Gemini's robustness)
- Correct temporal interpolation formula (Claude's precision)
- Automatic fallbacks and error recovery
- Memory-efficient processing
- Works with: LTX-2, SDXL, SD1.5, SVD, CogVideo
"""

import torch
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import gc


class UniversalSmartVAEDecode:
    """
    Universal VAE decoder with temporal sliding window.
    Handles both images and videos with automatic parameter detection.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "frames_per_batch": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Number of frames to decode at once. Lower = less VRAM, slower processing."
                }),
            },
            "optional": {
                "overlap_frames": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Temporal overlap for smoother transitions. Helps with boundary artifacts."
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Enabled",
                    "label_off": "Disabled",
                    "tooltip": "Enable spatial tiling for very high resolutions or low VRAM."
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Tile size in pixels (only used if tiling enabled)."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def __init__(self):
        self.last_vae_id = None
        self.cached_time_scale = None

    def detect_time_scale(self, vae, latents):
        """
        Detect temporal interpolation scale factor.
        Uses official VAE params if available, else auto-detects.
        """
        vae_id = id(vae)
        
        # Use cache if same VAE
        if vae_id == self.last_vae_id and self.cached_time_scale is not None:
            return self.cached_time_scale
        
        # Priority 1: Official VAE parameters
        if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula is not None:
            try:
                if isinstance(vae.downscale_index_formula, (list, tuple)) and len(vae.downscale_index_formula) >= 1:
                    time_scale = int(vae.downscale_index_formula[0])
                    print(f"🔍 Using VAE official time_scale: {time_scale}")
                    self.last_vae_id = vae_id
                    self.cached_time_scale = time_scale
                    return time_scale
            except:
                pass
        
        # Priority 2: Auto-detect with test sample
        try:
            # Use 3 frames for accurate detection
            # Formula: output = 1 + (input-1) * scale
            # So: 3 input → 1 + 2*scale output
            test_sample = latents[:, :, 0:3, :32, :32]
            
            with torch.no_grad():
                test_output = vae.decode(test_sample)
            
            test_output = self._normalize_output(test_output)
            output_frames = test_output.shape[0]
            
            # Solve: output_frames = 1 + (3-1) * scale
            # scale = (output_frames - 1) / 2
            detected_scale = max(1, (output_frames - 1) // 2)
            
            print(f"🔍 Auto-detected time_scale: {detected_scale} (3 frames → {output_frames} frames)")
            
            self.last_vae_id = vae_id
            self.cached_time_scale = detected_scale
            return detected_scale
        
        except Exception as e:
            print(f"⚠️  Scale detection failed: {e}. Using safe default: 1")
            return 1

    def _normalize_output(self, tensor):
        """
        Normalize VAE output to ComfyUI standard: [Frames, Height, Width, Channels]
        Handles various output formats from different VAE architectures.
        """
        # Handle list/tuple outputs (some VAEs return multiple items)
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        
        # 5D: [Batch, Channels, Frames, Height, Width]
        if tensor.dim() == 5:
            # Detect channel position
            if tensor.shape[1] in [3, 4]:
                # [B, C, F, H, W] → [B, F, H, W, C]
                tensor = tensor.permute(0, 2, 3, 4, 1)
            elif tensor.shape[-1] in [3, 4]:
                # Already [B, F, H, W, C]
                pass
            else:
                raise ValueError(f"Cannot determine channel position in 5D tensor: {tensor.shape}")
            
            # Flatten batch and frames: [B, F, H, W, C] → [B*F, H, W, C]
            b, f, h, w, c = tensor.shape
            tensor = tensor.reshape(b * f, h, w, c)
        
        # 4D: [Batch/Frames, Channels, Height, Width] or [Batch/Frames, Height, Width, Channels]
        elif tensor.dim() == 4:
            if tensor.shape[1] in [3, 4]:
                # [B, C, H, W] → [B, H, W, C]
                tensor = tensor.permute(0, 2, 3, 1)
            elif tensor.shape[-1] in [3, 4]:
                # Already [B, H, W, C]
                pass
            else:
                raise ValueError(f"Cannot determine channel position in 4D tensor: {tensor.shape}")
        
        else:
            raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}D with shape {tensor.shape}")
        
        # Clamp to valid range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # Trim to 3 channels if needed (some VAEs output 4)
        if tensor.shape[-1] > 3:
            tensor = tensor[..., :3]
        
        return tensor.contiguous()

    def _center_crop_to_reference(self, tensor, h_ref, w_ref):
        """
        Center crop tensor to match reference dimensions.
        Handles minor size differences from rounding during decode.
        """
        _, h, w, _ = tensor.shape
        
        if h == h_ref and w == w_ref:
            return tensor
        
        # Calculate crop offsets
        h_offset = max(0, (h - h_ref) // 2)
        w_offset = max(0, (w - w_ref) // 2)
        
        # Crop
        return tensor[:, h_offset:h_offset + h_ref, w_offset:w_offset + w_ref, :]

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, enable_tiling=False, tile_size=512):
        """
        Main decode function with sliding window and automatic parameter handling.
        """
        latents = samples["samples"]
        
        # ======== IMAGE PATH (4D) ========
        if latents.dim() == 4:
            print(f"🖼️  Image decode: {latents.shape}")
            
            with torch.no_grad():
                if enable_tiling and hasattr(vae, 'decode_tiled'):
                    output = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size)
                else:
                    output = vae.decode(latents)
            
            return (self._normalize_output(output),)
        
        # ======== VIDEO PATH (5D) ========
        batch, channels, total_frames, h_latent, w_latent = latents.shape
        
        # Validate and auto-correct parameters
        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        
        # Detect temporal scale
        time_scale = self.detect_time_scale(vae, latents)
        
        # Calculate expected output
        expected_frames = 1 + (total_frames - 1) * time_scale
        
        print(f"🎬 Video decode:")
        print(f"   Input: {total_frames} latent frames")
        print(f"   Time scale: {time_scale}x")
        print(f"   Expected output: {expected_frames} frames")
        print(f"   Batch size: {frames_per_batch}, Overlap: {overlap_frames}")
        
        # Initialize
        output_chunks = []
        h_reference = None
        w_reference = None
        
        pbar = comfy.utils.ProgressBar(total_frames)
        
        # ======== SLIDING WINDOW PROCESSING ========
        for start_idx in range(0, total_frames, frames_per_batch):
            throw_exception_if_processing_interrupted()
            
            end_idx = min(start_idx + frames_per_batch, total_frames)
            
            # Calculate context window (with overlap)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            
            # Extract latent chunk with context
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            # Decode chunk
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_chunk = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_chunk = vae.decode(latent_chunk)
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️  OOM detected. Falling back to tiled decode...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Retry with tiling
                    with torch.no_grad():
                        if hasattr(vae, 'decode_tiled'):
                            decoded_chunk = vae.decode_tiled(latent_chunk, tile_x=tile_size//2, tile_y=tile_size//2)
                        else:
                            # No tiled support - reduce batch and retry
                            print(f"   No tiled decode available. This may fail again...")
                            decoded_chunk = vae.decode(latent_chunk)
                else:
                    raise
            
            # Normalize output
            decoded_chunk = self._normalize_output(decoded_chunk)
            
            # ======== TEMPORAL TRIMMING ========
            # Remove overlap context, keep only the core frames
            front_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                # Last chunk: take everything from front_trim to end
                valid_frames = decoded_chunk[front_trim:]
            else:
                # Middle chunks: take exact core frames
                core_length = (end_idx - start_idx) * time_scale
                valid_frames = decoded_chunk[front_trim:front_trim + core_length]
            
            # ======== SPATIAL ALIGNMENT ========
            # Ensure all chunks have same spatial dimensions
            if h_reference is None:
                h_reference, w_reference = valid_frames.shape[1:3]
            else:
                valid_frames = self._center_crop_to_reference(valid_frames, h_reference, w_reference)
            
            output_chunks.append(valid_frames)
            
            # Memory cleanup
            del latent_chunk, decoded_chunk
            gc.collect()
            
            # Periodic CUDA cache clear (not every iteration for performance)
            if start_idx % (frames_per_batch * 3) == 0:
                torch.cuda.empty_cache()
            
            pbar.update(end_idx - start_idx)
        
        # ======== FINAL CONCATENATION ========
        final_output = torch.cat(output_chunks, dim=0)
        
        # Validation
        actual_frames = final_output.shape[0]
        print(f"✅ Decode complete: {actual_frames} frames")
        
        if abs(actual_frames - expected_frames) > time_scale:
            print(f"⚠️  Frame count warning:")
            print(f"   Expected: {expected_frames}")
            print(f"   Got: {actual_frames}")
            print(f"   Difference: {abs(actual_frames - expected_frames)}")
        
        return (final_output,)


# ======== NODE REGISTRATION ========
NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (Claude)",
}
