# ComfyUI-ImageForensics

**Image Forgery Detection and Localization for ComfyUI**

A ComfyUI custom node extension for image forensics, focused on detecting manipulated images and localizing potentially forged regions using deep learning models.

The project aims to integrate research-based image forgery detection methods into ComfyUI workflows, making image forensic analysis accessible through a modular node-based interface.

## Current Status

<badge>Experimental</badge>

Currently, the project includes a working integration of **TruFor**, with standalone inference and ComfyUI node execution successfully tested.

## Features

### TruFor Detector

Integration of the official [TruFor](https://github.com/grip-unina/TruFor) image forgery detection framework.

The custom node provides:

| Output | Type | Description |
|---|---|---|
| Forgery Map | IMAGE | Pixel-level manipulation prediction map |
| Confidence Map | IMAGE | Confidence map associated with the prediction |
| Detection Score | FLOAT | Image-level detection score |

The node supports ComfyUI IMAGE inputs and can process images through the standard ComfyUI workflow.

### Workflow Integration

The generated Forgery Map can be connected to standard ComfyUI nodes for further processing.

For example:

- `Mask to Image` for converting a mask into an image.
- `Image Composite Masked` for overlaying detected regions onto the original image.
- Other image and mask processing nodes for visualization and post-processing.

This allows users to build custom forensic analysis workflows without leaving ComfyUI.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DigitalAsocial/ComfyUI-ImageForensics.git
```

Restart ComfyUI.

The node will be available under:

```text
Image Forensics > TruFor Detector
```

## Model Setup

This project uses the official pretrained TruFor model and its required pretrained components.

Download the required model files from the official [TruFor repository](https://github.com/grip-unina/TruFor).

The expected directory structure is:

```text
ComfyUI-ImageForensics/
├── __init__.py
├── backends/
│   ├── __init__.py
│   └── trufor_backend.py
├── nodes/
│   ├── __init__.py
│   └── trufor_detector.py
└── TruFor/
    ├── config.py
    ├── data_core.py
    ├── trufor.yaml
    ├── models/
    ├── pretrained_models/
    │   ├── noiseprint++/
    │   │   └── noiseprint++.th
    │   └── segformers/
    │       └── mit_b2.pth
    └── weights/
        └── trufor.pth.tar
```

**Note:** Model weights are not included in this repository. Users must obtain them separately and place them in the appropriate directories.

## Dependencies

The integration relies on the ComfyUI PyTorch environment and the dependencies required by TruFor.

Main dependencies include:

- PyTorch
- torchvision
- timm
- yacs
- NumPy
- Pillow

Use the official TruFor repository for its complete dependency and model requirements.

Avoid installing or upgrading PyTorch independently inside an existing ComfyUI environment unless compatibility has been verified.

## Technical Notes

- The implementation uses the official TruFor model architecture and pretrained checkpoints.
- Inference is performed through a dedicated backend module.
- Model loading is cached to avoid reloading the network for every inference.
- Input and output tensors follow ComfyUI image conventions.
- The Forgery Map represents pixel-level predictions; it should not be interpreted as a definitive proof of manipulation.
- The Detection Score is the model's image-level output and should not be treated as a calibrated probability without further validation.

## Roadmap

Planned development:

- [x] Integrate TruFor backend.
- [x] Load official pretrained weights.
- [x] Validate standalone inference.
- [x] Implement TruFor ComfyUI custom node.
- [x] Generate Forgery Map and Confidence Map.
- [x] Generate image-level Detection Score.
- [x] Validate mask-based post-processing in ComfyUI.
- [ ] Add configurable forgery threshold.
- [ ] Add direct ComfyUI MASK output.
- [ ] Integrate MVSS-Net.
- [ ] Build a unified multi-model forensic workflow.
- [ ] Add comparison and visualization utilities.

## Disclaimer

This project is intended for research, experimentation, and image forensic analysis.

Deep learning-based forgery detection is not infallible. Predictions may contain false positives or false negatives, and results should be interpreted alongside other forensic evidence.

## Acknowledgements

This project builds upon the research and implementation of:

**TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization**

Original implementation: [grip-unina/TruFor](https://github.com/grip-unina/TruFor)

All credit for the original TruFor architecture, methodology, and pretrained models belongs to its respective authors.

## License

This repository's licensing should be considered separately from the licenses of the integrated research models and their pretrained weights.

Refer to the original TruFor repository for its licensing terms.
