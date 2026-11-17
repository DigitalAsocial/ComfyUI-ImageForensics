
import torch
import comfy.model_management as mm

from ..backends.trufor_backend import run_trufor


def gray_to_comfy_image(x):
    """
    Input:  [B, 1, H, W]
    Output: [B, H, W, 3]
    """

    x = x.clamp(0, 1)

    x = x.permute(0, 2, 3, 1)

    x = x.repeat(1, 1, 1, 3)

    return x.float()


class TruForDetector:

    CATEGORY = "Image Forensics"

    FUNCTION = "detect"

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "FLOAT",
    )

    RETURN_NAMES = (
        "forgery_map",
        "confidence_map",
        "score",
    )

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def detect(self, image):

        device = mm.get_torch_device()

        # ComfyUI: [B,H,W,C]
        # TruFor:  [B,C,H,W]

        rgb = image.permute(0, 3, 1, 2).contiguous()

        results = run_trufor(
            rgb,
            device=device
        )

        forgery_map = gray_to_comfy_image(
            results["forgery_map"]
        )

        confidence_map = gray_to_comfy_image(
            results["confidence_map"]
        )

        # Current node interface returns one FLOAT.
        # For batch input, use the mean score.
        score = float(
            results["score"].mean().item()
        )

        return (
            forgery_map,
            confidence_map,
            score,
        )


NODE_CLASS_MAPPINGS = {
    "TruForDetector": TruForDetector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TruForDetector": "TruFor Detector"
}
