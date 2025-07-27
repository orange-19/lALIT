import pdfplumber
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import os

class LalitHeadingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def extract_features_from_pdf(pdf_path):
    """
    Extracts features and text for each line in the PDF.
    Returns: features (np.ndarray), texts (list), pages (list)
    """
    features = []
    texts = []
    pages = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                lines = page.extract_text_lines(extra_attrs=["size", "fontname"])
                for line in lines:
                    # Handle missing 'size' key with default value
                    font_size = line.get('size', 12.0)  # Default font size
                    font_name = line.get('fontname', '')
                    text = line.get('text', '').strip()
                    top = line.get('top', 0)
                    
                    # Skip empty lines
                    if not text:
                        continue
                        
                    length = len(text)
                    # Encode font name as a simple int (hash), or use one-hot in advanced cases
                    font_hash = hash(font_name) % 1000
                    features.append([font_size, top, length, font_hash])
                    texts.append(text)
                    pages.append(page_number)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        raise
        
    if not features:
        raise ValueError(f"No text features extracted from PDF: {pdf_path}")
        
    return np.array(features), texts, pages

def label_headings_by_font_size(features, texts, num_levels=3):
    """
    Improved rule-based labeling: assign heading levels by font size and text characteristics.
    Returns: labels (H1=0, H2=1, H3=2, normal=3)
    """
    font_sizes = features[:,0]
    text_lengths = features[:,2]
    
    # Filter out very short texts and numbers-only texts that are likely not headings
    labels = []
    for i, (fs, text) in enumerate(zip(font_sizes, texts)):
        # Skip very short texts (likely not headings)
        if len(text.strip()) < 3:
            labels.append(num_levels)  # Normal text
            continue
            
        # Skip texts that are just numbers or symbols
        if text.strip().replace('.', '').replace(')', '').replace('(', '').isdigit():
            labels.append(num_levels)  # Normal text
            continue
            
        # Skip texts that are mostly dots or symbols
        if len([c for c in text if c in '.,;:()[]{}']) > len(text) * 0.5:
            labels.append(num_levels)  # Normal text
            continue
            
        labels.append(0)  # Consider as potential heading initially
    
    # Now assign proper heading levels based on font size
    heading_indices = [i for i, label in enumerate(labels) if label == 0]
    if heading_indices:
        heading_font_sizes = [font_sizes[i] for i in heading_indices]
        unique_sizes = sorted(set(heading_font_sizes), reverse=True)
        
        # Create size to level mapping for actual headings
        size_to_level = {}
        for i, size in enumerate(unique_sizes[:num_levels]):
            size_to_level[size] = i
        
        # Assign proper levels to headings
        for i in heading_indices:
            fs = font_sizes[i]
            labels[i] = size_to_level.get(fs, min(num_levels-1, len(unique_sizes)-1))
    
    return np.array(labels)

def train_lalit_model(X, y, input_dim, model_path="lalit_heading_model.pth", epochs=15):
    # Get unique classes and remap labels to continuous range starting from 0
    unique_classes = sorted(set(y))
    num_classes = len(unique_classes)
    
    # Create mapping from original labels to continuous range [0, num_classes-1]
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    y_remapped = np.array([class_mapping[label] for label in y])
    
    print(f"Training model with {input_dim} input features and {num_classes} classes")
    print(f"Class mapping: {class_mapping}")
    
    model = LalitHeadingModel(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_remapped, dtype=torch.long)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    # Save both model state and metadata
    model_data = {
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'class_mapping': class_mapping,
        'unique_classes': unique_classes
    }
    torch.save(model_data, model_path)
    print(f"Model saved to {model_path} ({os.path.getsize(model_path)/1024/1024:.2f} MB)")
    return model

def predict_headings(pdf_path, model_path, num_levels=3):
    features, texts, pages = extract_features_from_pdf(pdf_path)
    input_dim = features.shape[1]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using rule-based approach.")
        # Fallback to rule-based labeling
        labels = label_headings_by_font_size(features, texts, num_levels)
        level_map = {0: "H1", 1: "H2", 2: "H3"}
        outline = []
        for idx, label in enumerate(labels):
            if label in level_map:
                outline.append({
                    "level": level_map[label],
                    "text": texts[idx],
                    "page": pages[idx]
                })
        return outline
    
    try:
        # Load model with metadata
        model_data = torch.load(model_path, map_location='cpu')
        
        # Handle both old and new model formats
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            saved_input_dim = model_data['input_dim']
            saved_num_classes = model_data['num_classes']
            state_dict = model_data['state_dict']
            # Get class mapping if available
            class_mapping = model_data.get('class_mapping', {})
            unique_classes = model_data.get('unique_classes', list(range(saved_num_classes)))
        else:
            # Old format - try to infer from state dict
            state_dict = model_data
            saved_input_dim = input_dim
            saved_num_classes = num_levels + 1
            class_mapping = {}
            unique_classes = list(range(saved_num_classes))
        
        model = LalitHeadingModel(input_dim=saved_input_dim, num_classes=saved_num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(features, dtype=torch.float32)
            pred = torch.argmax(model(X_tensor), axis=1).numpy()
        
        # Map predictions back to original class labels if class mapping exists
        if class_mapping:
            # Create reverse mapping
            reverse_mapping = {v: k for k, v in class_mapping.items()}
            pred_original = [reverse_mapping.get(p, p) for p in pred]
        else:
            pred_original = pred
            
        # Map labels to H1/H2/H3/Normal
        level_map = {0: "H1", 1: "H2", 2: "H3"}
        outline = []
        for idx, p in enumerate(pred_original):
            if p in level_map:
                outline.append({
                    "level": level_map[p],
                    "text": texts[idx],
                    "page": pages[idx]
                })
        return outline
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        print("Falling back to rule-based approach.")
        # Fallback to rule-based labeling
        labels = label_headings_by_font_size(features, texts, num_levels)
        level_map = {0: "H1", 1: "H2", 2: "H3"}
        outline = []
        for idx, label in enumerate(labels):
            if label in level_map:
                outline.append({
                    "level": level_map[label],
                    "text": texts[idx],
                    "page": pages[idx]
                })
        return outline

def main():
    parser = argparse.ArgumentParser(description="PDF Heading Detection with Lalit Model")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--train", action="store_true", help="Train model on this PDF")
    parser.add_argument("--model", type=str, default="lalit_heading_model.pth", help="Path to save/load model")
    parser.add_argument("--json_out", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    # Validate PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return 1

    try:
        # Step 1: Extract features and labels
        features, texts, pages = extract_features_from_pdf(args.pdf_path)
        input_dim = features.shape[1]
        labels = label_headings_by_font_size(features, texts)

        # Step 2: Train model if requested
        if args.train:
            train_lalit_model(features, labels, input_dim, model_path=args.model)

        # Step 3: Predict headings and print/save JSON
        outline = predict_headings(args.pdf_path, args.model)
        
        if args.json_out:
            try:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(outline, f, indent=2, ensure_ascii=False)
                print(f"Results saved to {args.json_out}")
            except Exception as e:
                print(f"Error saving to {args.json_out}: {str(e)}")
                return 1
        else:
            print(json.dumps(outline, indent=2, ensure_ascii=False))
            
        return 0
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
