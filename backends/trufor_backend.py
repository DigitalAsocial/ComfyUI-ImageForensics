
import os
import sys

import torch
from torch.nn import functional as F


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRUFOR_DIR = os.path.join(
    ROOT_DIR,
    "TruFor"
)

WEIGHTS_PATH = os.path.join(
    TRUFOR_DIR,
    "weights",
    "trufor.pth.tar"
)

CONFIG_PATH = os.path.join(
    TRUFOR_DIR,
    "trufor.yaml"
)

# Allow TruFor internal imports such as:
# from models.DnCNN import make_net

if TRUFOR_DIR not in sys.path:
    sys.path.insert(0, TRUFOR_DIR)


# --------------------------------------------------
# Model cache
# --------------------------------------------------

_MODEL = None
_DEVICE = None



def load_config():

    from config import _C as config

    config.defrost()

    config.merge_from_file(CONFIG_PATH)

    # SegFormer-B2 pretrained weights
    config.MODEL.PRETRAINED = os.path.join(
        TRUFOR_DIR,
        "pretrained_models",
        "segformers",
        "mit_b2.pth"
    )

    # Noiseprint++ pretrained weights
    config.MODEL.EXTRA.NP_WEIGHTS = os.path.join(
        TRUFOR_DIR,
        "pretrained_models",
        "noiseprint++",
        "noiseprint++.th"
    )

    config.freeze()

    return config



def get_model(device):

    global _MODEL
    global _DEVICE

    device = torch.device(device)

    if _MODEL is not None and _DEVICE == device:
        return _MODEL

    # Release previous cached model if device changed.
    _MODEL = None
    _DEVICE = None

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"TruFor weights not found:\n{WEIGHTS_PATH}"
        )

    config = load_config()

    from models.cmx.builder_np_conf import myEncoderDecoder

    model = myEncoderDecoder(cfg=config)

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()
    model.to(device)

    _MODEL = model
    _DEVICE = device

    return _MODEL


# --------------------------------------------------
# Inference
# --------------------------------------------------

@torch.inference_mode()
def run_trufor(image, device):

    model = get_model(device)

    image = image.to(device=device, dtype=torch.float32)

    # ComfyUI IMAGE range [0,1].
    # Official TruFor dataset divides 8-bit RGB by 256.
    image = image * (255.0 / 256.0)

    pred, conf, det, npp = model(image)

    score = torch.sigmoid(det).reshape(-1)

    # Localization map: foreground / manipulation probability.
    pred = F.softmax(pred, dim=1)[:, 1:2, :, :]

    # Confidence map.
    conf = torch.sigmoid(conf)

    return {
        "forgery_map": pred.detach().cpu(),
        "confidence_map": conf.detach().cpu(),
        "score": score.detach().cpu(),
        "noiseprint": npp.detach().cpu()
    }
