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
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_file_from_hf(filename):
    return hf_hub_download(
        repo_id="weakyy/image-captioning-baseline-model",
        filename=filename,
        repo_type="model"
    )

# Model loading from Hugging Face
def load_baseline_model():
    # Load vocabulary
    with open(download_file_from_hf("word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)
    with open(download_file_from_hf("idx2word.pkl"), "rb") as f:
        idx2word = pickle.load(f)

    vocab = {
        "word2idx": word2idx,
        "idx2word": idx2word
    }

    vocab_size = len(vocab["word2idx"])
    
    encoder = EncoderCNN().eval()
    decoder = DecoderRNN(attention_dim=256, embed_dim=256, decoder_dim=512, vocab_size=vocab_size).eval()
    
    # Corrected weight loading
    encoder.load_state_dict(torch.load(download_file_from_hf("encoder.pth"), map_location=device))
    decoder.load_state_dict(torch.load(download_file_from_hf("decoder.pth"), map_location=device), strict=False)
    
    return encoder, decoder, vocab

def generate_baseline_caption(image_tensor, encoder, decoder, vocab, beam_size=3, max_len=20):
    encoder_out = encoder(image_tensor)
    encoder_dim = encoder_out.size(-1)
    num_pixels = encoder_out.size(1)
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)

    seqs = torch.full((beam_size, 1), vocab['word2idx']['<start>'],
                      dtype=torch.long, device=image_tensor.device)
    top_k_scores = torch.zeros(beam_size, 1, device=image_tensor.device)
    complete_seqs = []
    complete_seqs_scores = []
    complete_seqs_alphas = []
    h, c = decoder.init_hidden_state(encoder_out.mean(dim=1))

    for step in range(max_len):
        prev_words = seqs[:, -1].unsqueeze(1)
        embeddings = decoder.embedding(prev_words)
        context, alpha = decoder.attention(encoder_out, h)
        lstm_input = torch.cat([embeddings.squeeze(1), context], dim=1)
        h, c = decoder.decode_step(lstm_input, (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 0:
            top_k_scores, top_k_words = scores[0].topk(beam_size, dim=0)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size, dim=0)

        prev_seq_inds = top_k_words // len(vocab['idx2word'])
        next_word_inds = top_k_words % len(vocab['idx2word'])
        seqs = torch.cat([seqs[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, word in enumerate(next_word_inds)
                           if word != vocab['word2idx']['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if complete_inds:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
            complete_seqs_alphas.extend([alpha[i].cpu().numpy() for i in complete_inds])

        if len(incomplete_inds) == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_seq_inds[incomplete_inds]]
        c = c[prev_seq_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_seq_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_seqs_scores = top_k_scores.tolist()
        complete_seqs_alphas = alpha.cpu().numpy().tolist()

    best_idx = np.argmax(complete_seqs_scores)
    caption_ids = complete_seqs[best_idx]
    alphas = complete_seqs_alphas[best_idx] if complete_seqs_alphas else []

    caption_words = []
    for word_id in caption_ids:
        word = vocab['idx2word'][word_id]
        if word not in ['<start>', '<end>', '<pad>']:
            caption_words.append(word)
        if word == '<end>':
            break

    confidence = min(float(np.exp(np.max(complete_seqs_scores))), 1.0)
    return {
        "caption": ' '.join(caption_words),
        "confidence": confidence,
        "alphas": alphas
    }

def enhance_with_openai(caption, max_tokens=100, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Improve this image caption concisely while preserving factual accuracy:"},
                {"role": "user", "content": caption}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return None

def load_image(image_source):
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

def preprocess_image(image, device=device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)
