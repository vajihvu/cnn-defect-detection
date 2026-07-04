import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from data_setup import get_dataloaders
import argparse
import os

def evaluate_model(model_name, threshold=0.5):

    _, val_ds, _, _, _ = get_dataloaders(batch_size=32)
    

    model_path = f'saved_models/{model_name}_best.keras'
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Please train first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"Evaluating {model_name}...")
    

    for inputs, labels in val_ds:
        probs = model.predict(inputs, verbose=0)
        preds = (probs >= threshold).astype(int)
        
        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.numpy().flatten())
        
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n--- Evaluation Metrics (Defect Class) ---")
    print(f"Precision: {precision:.4f} (When model says 'Defect', how often is it right?)")
    print(f"Recall:    {recall:.4f} (Of all real defects, how many did the model catch?)")
    print(f"F1 Score:  {f1:.4f}")
    

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Defect'], yticklabels=['Good', 'Defect'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    print(f"\nSaved confusion matrix plot to {model_name}_confusion_matrix.png")
    

    precisions, recalls, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name}_pr_curve.png')
    print(f"Saved PR curve plot to {model_name}_pr_curve.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnet', choices=['baseline', 'efficientnet'])
    args = parser.parse_args()
    
    evaluate_model(args.model)
