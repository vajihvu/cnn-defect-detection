# Aircraft Component Defect Detection

This project builds a Convolutional Neural Network (CNN) based visual inspection system to detect surface defects on aerospace and manufacturing components. 

It was built to demonstrate a transition from classical ML techniques to Deep Learning, specifically showcasing:
1. **Transfer Learning**: Utilizing a ResNet18 backbone pre-trained on ImageNet to extract complex features, proving more effective than training a CNN from scratch on limited data.
2. **Handling Class Imbalance**: In real-world manufacturing, defects are rare. This project addresses the severe imbalance between "good" and "defect" samples using weighted loss functions (`BCEWithLogitsLoss(pos_weight)`) and data augmentation.
3. **Targeted Evaluation Metrics**: Moving beyond raw accuracy, the model is evaluated on **Precision and Recall** specifically for the minority 'defect' class, mirroring evaluation strategies used in fraud detection and other imbalanced domains.

## Dataset Setup

The code is designed to work with the **MVTec AD Dataset** (specifically the `metal_nut` or `screw` categories).

1. Download the `metal_nut` category from the MVTec AD dataset.
2. Extract the `metal_nut` folder into the root of this project.
3. Run the setup script to reorganize the raw data into a supervised binary classification format:

```bash
python data_setup.py
```

This will create a `dataset/` directory with `train/` and `val/` splits, separating the images into `good` and `defect` classes.

## Training

You can train either a simple baseline CNN built from scratch, or the industry-standard ResNet18 transfer learning model.

To train the ResNet model:
```bash
python train.py --model resnet --epochs 10
```

To train the Baseline model (for comparison):
```bash
python train.py --model baseline --epochs 10
```

The script automatically calculates class weights to penalize the model more heavily for missing a defect, and saves the best model weights to the `saved_models/` directory.

## Evaluation

To evaluate the trained model on the validation set and generate Precision/Recall metrics, a Confusion Matrix, and a PR Curve:

```bash
python evaluate.py --model resnet
```

*(Note: In an industrial setting, minimizing False Negatives—missing a defect—is critical, so we aim for high Recall).*

## Deployment (Web Demo)

The project includes a Streamlit web application to easily demonstrate the model inference on new images.

```bash
streamlit run app.py
```

Upload an image of a component, and the app will output a Confidence Score and classify the part as either "PASSED" or "DEFECT DETECTED".
