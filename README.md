# Project Computervision WS24/25 - Binary masks for rain dataset

The goal of this project is to generate binary masks for the rain drops in the dataset.
The dataset consists of images with (raindrops appear in bright white) and without flash (raindrops appear grey-ish).

## Guidelines
Here are some ideas for binary mask creation.

### RGB color space
1. **Color Thresholding**: Rain streaks appear bright. We can threshold with high RGB values.<br>
Problem: Background also gets detected
2. **Highlighting Features**: We can try using contrast enhancements to make bright areas even brighter and dark areas even darker. To enhance contrast we can use
  - Histogram equalization
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)

### HSV color space
1. **Brightness Isolation (V Channel)**: Value/Brightness channel in HSV could also be used for thresholding, because the rain streaks are bright in the flash (set hight V threshold).
2. **Saturation Filtering (S Channel)**: Rain streaks have usually lower saturation (because they are nearly white). The S channel can be used with a low threshold.

## Further Steps
Once the masks are generated, a NERF (Nerfacto in Nerfstudio) is trained both without the masks and with the masks for both the flash and no flash datasets.
Code for training NERF without masks:
```
conda activate nerfstudio
ns-train nerfacto --vis viewer --data "data_path" --pipeline.model.camera-optimizer.mode off
nerfstudio-data --downscale-factor 1 --eval_mode interval --output-dir "output_path" --max-num-iterations = 30000
```
Code for loading config to train NERF with masks:
```
ns-export nerfacto --load-config outputs\path\config.yml --output-dir exports/output/
ns-viewer --load-config "output_path\config.yml" --vis viewer
```
Code for training NERF with masks (maybe masks need to be specified in transforms.json for each image):
```
ns-train splatfacto --vis viewer --colormap --data data/dataset --masks-path masks --downscale-factor 1
```
