import torch
import torch.nn.functional as F
import openai
import numpy as np
from PIL import Image
import requests
import pickle
from io import BytesIO
from models import EncoderCNN, DecoderRNN
from huggingface_hub import hf_hub_download


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_file_from_hf(filename):
    return hf_hub_download(
        repo_id="weakyy/image-captioning-baseline-model",
        filename=filename,
        repo_type="model"
    )
# Model loading from Hugging Face
def load_baseline_model():
    model_files = {
        "encoder": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/encoder.pth",
        "decoder": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/decoder.pth",
        "word2idx": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/word2idx.pkl",
        "idx2word": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/idx2word.pkl"
    }
    # Load vocabulary
    with open(download_file_from_hf("word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)
    with open(download_file_from_hf("idx2word.pkl"), "rb") as f:
        idx2word = pickle.load(f)

    vocab = {
    "word2idx": word2idx,
    "idx2word": idx2word
}

    
    # Now safe to get vocab_size
    vocab_size = len(vocab["word2idx"])
    
    # Initialize model architecture
    encoder = EncoderCNN().eval()
    decoder = DecoderRNN(attention_dim=256, embed_dim=256, decoder_dim=512, vocab_size=vocab_size).eval()
    
    # Download and load weights
    encoder.load_state_dict(torch.load(download_file_from_hf("encoder.pth"), strict=False, map_location=device))

    decoder.load_state_dict(torch.load(download_file_from_hf("decoder.pth"), strict=False, map_location=device))
    
        
    return encoder, decoder, vocab

def generate_baseline_caption(image_tensor, encoder, decoder, vocab, beam_size=3, max_len=20):
    """
    Generate caption using beam search with attention
    Args:
        image_tensor: Processed image tensor (1, 3, H, W)
        encoder: CNN encoder model
        decoder: RNN decoder with attention
        vocab: Dictionary containing 'word2idx' and 'idx2word'
        beam_size: Number of beams to use (default: 3)
        max_len: Maximum caption length (default: 20)
    Returns:
        {"caption": str, "confidence": float, "alphas": np.array}
    """
    # Encode the image
    encoder_out = encoder(image_tensor)  # (1, num_pixels, encoder_dim)
    encoder_dim = encoder_out.size(-1)
    num_pixels = encoder_out.size(1)
    
    # Expand for beam search
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    
    # Initialize sequences and scores
    seqs = torch.full((beam_size, 1), vocab['word2idx']['<start>'], 
                     dtype=torch.long, device=image_tensor.device)  # (k, 1)
    top_k_scores = torch.zeros(beam_size, 1, device=image_tensor.device)
    
    # Storage for completed sequences
    complete_seqs = []
    complete_seqs_scores = []
    complete_seqs_alphas = []
    
    # Initialize hidden state
    h, c = decoder.init_hidden_state(encoder_out.mean(dim=1))  # (k, decoder_dim)
    
    # Beam search
    for step in range(max_len):
        prev_words = seqs[:, -1].unsqueeze(1)  # (k, 1)
        
        # Get embeddings and attention
        embeddings = decoder.embedding(prev_words)  # (k, 1, embed_dim)
        context, alpha = decoder.attention(encoder_out, h)  # (k, encoder_dim), (k, num_pixels)
        
        # LSTM step
        lstm_input = torch.cat([embeddings.squeeze(1), context], dim=1)  # (k, embed_dim + encoder_dim)
        h, c = decoder.decode_step(lstm_input, (h, c))
        
        # Predict next words
        scores = F.log_softmax(decoder.fc(h), dim=1)  # (k, vocab_size)
        scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)
        
        # Get top k candidates
        if step == 0:
            top_k_scores, top_k_words = scores[0].topk(beam_size, dim=0)  # (k)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size, dim=0)  # (k)
        
        # Convert to sequence indices
        prev_seq_inds = top_k_words // len(vocab['idx2word'])  # (k)
        next_word_inds = top_k_words % len(vocab['idx2word'])  # (k)
        
        # Update sequences
        seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)
        
        # Track completed sequences
        incomplete_inds = [ind for ind, word in enumerate(next_word_inds) 
                          if word != vocab['word2idx']['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        if complete_inds:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
            complete_seqs_alphas.extend([alpha[i].cpu().numpy() for i in complete_inds])
        
        # Stop if all beams complete
        if len(incomplete_inds) == 0:
            break
            
        # Continue with incomplete sequences
        seqs = seqs[incomplete_inds]
        h = h[prev_seq_inds[incomplete_inds]]
        c = c[prev_seq_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_seq_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
    
    # Handle case where no sequence completed
    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_seqs_scores = top_k_scores.tolist()
        complete_seqs_alphas = alpha.cpu().numpy().tolist()
    
    # Select best sequence
    best_idx = np.argmax(complete_seqs_scores)
    caption_ids = complete_seqs[best_idx]
    alphas = complete_seqs_alphas[best_idx] if complete_seqs_alphas else []
    
    # Convert to words
    caption_words = []
    for word_id in caption_ids:
        word = vocab['idx2word'][word_id]
        if word not in ['<start>', '<end>', '<pad>']:
            caption_words.append(word)
        if word == '<end>':
            break
    
    # Calculate confidence (normalized score)
    confidence = min(float(np.exp(np.max(complete_seqs_scores))), 1.0)
    
    return {
        "caption": ' '.join(caption_words),
        "confidence": confidence,
        "alphas": alphas
    }

def enhance_with_openai(caption, max_tokens=100, temperature=0.7):
    """
    Enhance caption using GPT-3.5
    Args:
        caption: Original caption text
        max_tokens: Maximum response length
        temperature: Creativity control (0-1)
    Returns:
        Enhanced caption (str) or None if failed
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Improve this image caption concisely while preserving factual accuracy:"
                },
                {
                    "role": "user",
                    "content": caption
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return None

def load_image(image_source):
    """
    Safely load image from file or URL
    Args:
        image_source: File path, URL, or file-like object
    Returns:
        PIL.Image or None if failed
    """
    try:
        if isinstance(image_source, str):
            if image_source.startswith(('http:', 'https:')):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                return Image.open(image_source).convert('RGB')
        else:
            return Image.open(image_source).convert('RGB')
    except Exception as e:
        print(f"Image load failed: {str(e)}")
        return None

def preprocess_image(image, device='cpu'):
    """
    Transform image for model input
    Args:
        image: PIL.Image
        device: Target device
    Returns:
        torch.Tensor (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)
