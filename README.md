# Image-Classification-using-EfficientNet

This project uses a pre-trained EfficientNet model to classify images into different categories. The model is fine-tuned on a custom dataset for this purpose.

### Note:

- I added a new section under "Data" specifically for the ImageNet dataset.
- Provided a link to the ImageNet download page. Users will need to register and follow the instructions on the ImageNet website to download the dataset.
- Emphasized the need to organize the downloaded dataset into `train` and `test` folders within the `data` directory of your project.

## Data

### ImageNet Dataset

You can download the ImageNet dataset using the following links:

- [ImageNet Download (Registration Required)](http://www.image-net.org/download)

After downloading, extract the dataset and organize it into `train` and `test` folders as per your project's data structure.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/SreeEswaran/Image-Classification-using-EfficientNet.git
cd Image-Classification-using-EfficientNet
```
2. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Training the model
    ```bash
    python scripts/train.py
    ```
4. Evaluating the model
   ```bash
   python scripts/evaluate.py
   ```
5. Making predictions
   ```bash
   python scripts/predict.py
   ```


