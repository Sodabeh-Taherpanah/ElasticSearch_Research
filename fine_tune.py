from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import json
import torch
import os

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def simple_fine_tune():
    # Set device for Apple Silicon
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Using device: {device}")
    
    # Prepare your training data
    train_samples = []
    
    # Load your data
    with open("professor_dataset.jsonl", 'r') as f:
        prof_data = json.loads(f.readline().strip())
    
    query = prof_data["question"]
    relevant_doc_ids = set(prof_data["expected_resources"])
    
    # Load corpus
    corpus = {}
    with open("corpus.jsonl", 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc["_id"]] = doc["text"]
    
    # Create training samples
    for doc_id in relevant_doc_ids:
        if doc_id in corpus:
            doc_text = corpus[doc_id][:1000]  # Shorter text for Mac
            train_samples.append(InputExample(
                texts=[query, doc_text], 
                label=1.0
            ))
    
    # Negative examples
    all_doc_ids = list(corpus.keys())
    irrelevant_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in relevant_doc_ids]
    
    for doc_id in irrelevant_doc_ids[:len(train_samples)]:
        doc_text = corpus[doc_id][:1000]  # Shorter text for Mac
        train_samples.append(InputExample(
            texts=[query, doc_text], 
            label=0.0
        ))
    
    print(f"📊 Training samples: {len(train_samples)}")
    
    # Create model
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Train the model
    print("🔥 Starting fine-tuning...")
    model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=8),
        epochs=10,
        show_progress_bar=True
    )
    
    # CRITICAL: MANUAL SAVE - This was missing!
    print("💾 Saving model to 'fine_tuned_biomedical_cross_encoder' folder...")
    model.save('fine_tuned_biomedical_cross_encoder')
    print("✅ Model saved successfully!")
    
    # Verify the folder was created
    import glob
    if os.path.exists('fine_tuned_biomedical_cross_encoder'):
        files = glob.glob('fine_tuned_biomedical_cross_encoder/*')
        print(f"📁 Folder created with {len(files)} files")
    else:
        print("❌ Folder still not created - checking permissions...")
        # Try different directory
        model.save('./fine_tuned_biomedical_cross_encoder')
        print("💾 Trying current directory...")

if __name__ == "__main__":
    simple_fine_tune()