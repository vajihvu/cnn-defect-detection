import tensorflow as tf
from data_setup import get_dataloaders
from models import BaselineCNN, TransferLearningCNN
import argparse
import os

def train_model(model_name, epochs=10, batch_size=32, lr=0.001):

    os.makedirs('saved_models', exist_ok=True)

    train_ds, val_ds, class_names, num_defect, num_good = get_dataloaders(batch_size=batch_size)
    print(f"Training on {num_good} 'good' images and {num_defect} 'defect' images.")

    weight_for_good = 1.0
    weight_for_defect = float(num_good) / max(num_defect, 1)
    class_weight = {0: weight_for_good, 1: weight_for_defect}
    
    print(f"Imbalance class weights: Good (0) -> {weight_for_good:.2f}, Defect (1) -> {weight_for_defect:.2f}")

    if model_name == 'baseline':
        model = BaselineCNN()
    elif model_name == 'efficientnet':
        model = TransferLearningCNN(freeze_backbone=True)
    else:
        raise ValueError("Invalid model name")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    save_path = f'saved_models/{model_name}_best.keras'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    print(f"\nStarting training for {model_name}...")
    

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[checkpoint_callback],
        verbose=1
    )

    print(f"Training complete. Best model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnet', choices=['baseline', 'efficientnet'])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train_model(model_name=args.model, epochs=args.epochs)
