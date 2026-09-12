"""
Flask front end for the Urdu Question Generation model.

This file rebuilds the exact same layers used in the training notebook
(same names, same shapes) so that torch.load(...).load_state_dict(...)
lines up correctly with best_model.pt.

Folder layout expected:
    urdu_qg_app/
        app.py              <- this file
        best_model.pt       <- copy from your notebook
        ur_sp.model         <- copy from your notebook
        templates/
            index.html
"""

import os
import torch
import torch.nn as nn
import sentencepiece as spm
from flask import Flask, render_template, request

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_model.pt")
SP_MODEL_PATH = os.path.join(BASE_DIR, "ur_sp.model")

ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"
MAX_LEN = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)
PAD, UNK, BOS, EOS = 0, 1, 2, 3
vocab_size = sp.get_piece_size()

# ---------------------------------------------------------------------
# Rebuild the exact same architecture as the notebook
# ---------------------------------------------------------------------
embedding = nn.Embedding(vocab_size, 256, padding_idx=PAD)

encoder = nn.LSTM(
    input_size=256,
    hidden_size=512,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3,
)

decoder_embedding = nn.Embedding(vocab_size, 256, padding_idx=PAD)

decoder = nn.LSTM(
    input_size=1280,   # 256 (embedding) + 1024 (bidirectional context)
    hidden_size=512,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
)

output_layer = nn.Linear(512, vocab_size)

attention_W1 = nn.Linear(1024, 512)
attention_W2 = nn.Linear(512, 512)
attention_V = nn.Linear(512, 1)

hidden_fc = nn.Linear(512, 512)
cell_fc = nn.Linear(512, 512)

# Move everything to device
embedding = embedding.to(device)
encoder = encoder.to(device)
decoder_embedding = decoder_embedding.to(device)
decoder = decoder.to(device)
output_layer = output_layer.to(device)
attention_W1 = attention_W1.to(device)
attention_W2 = attention_W2.to(device)
attention_V = attention_V.to(device)
hidden_fc = hidden_fc.to(device)
cell_fc = cell_fc.to(device)

# ---------------------------------------------------------------------
# Load trained weights
# ---------------------------------------------------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

embedding.load_state_dict(checkpoint["embedding"])
encoder.load_state_dict(checkpoint["encoder"])
decoder_embedding.load_state_dict(checkpoint["decoder_embedding"])
decoder.load_state_dict(checkpoint["decoder"])
output_layer.load_state_dict(checkpoint["output_layer"])
attention_W1.load_state_dict(checkpoint["attention_W1"])
attention_W2.load_state_dict(checkpoint["attention_W2"])
attention_V.load_state_dict(checkpoint["attention_V"])
hidden_fc.load_state_dict(checkpoint["hidden_fc"])
cell_fc.load_state_dict(checkpoint["cell_fc"])

embedding.eval()
encoder.eval()
decoder_embedding.eval()
decoder.eval()
output_layer.eval()
attention_W1.eval()
attention_W2.eval()
attention_V.eval()
hidden_fc.eval()
cell_fc.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(valid_loss={checkpoint['valid_loss']:.4f})")


# ---------------------------------------------------------------------
# Attention (identical to the notebook)
# ---------------------------------------------------------------------
def attention(decoder_hidden, encoder_outputs, mask=None):
    decoder_hidden = decoder_hidden.unsqueeze(1)

    score = attention_V(
        torch.tanh(
            attention_W1(encoder_outputs)
            + attention_W2(decoder_hidden)
        )
    )  # [batch, src_len, 1]

    if mask is not None:
        score = score.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

    weights = torch.softmax(score, dim=1)
    context = torch.sum(weights * encoder_outputs, dim=1)

    return context, weights


# ---------------------------------------------------------------------
# Shared encoder pass
# ---------------------------------------------------------------------
def _encode_source(src_text):
    src_ids = [BOS] + sp.encode(src_text, out_type=int) + [EOS]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = (src_tensor != PAD)

    src_embedded = embedding(src_tensor)
    encoder_outputs, (hidden, cell) = encoder(src_embedded)

    hidden = hidden.view(2, 2, 1, 512).sum(dim=1)
    cell = cell.view(2, 2, 1, 512).sum(dim=1)

    decoder_hidden = torch.stack([hidden_fc(hidden[0]), hidden_fc(hidden[1])])
    decoder_cell = torch.stack([cell_fc(cell[0]), cell_fc(cell[1])])

    return encoder_outputs, src_mask, decoder_hidden, decoder_cell


# ---------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_question(src_text, max_len=MAX_LEN):
    encoder_outputs, src_mask, decoder_hidden, decoder_cell = _encode_source(src_text)

    decoder_input = torch.tensor([BOS], dtype=torch.long).to(device)
    generated_ids = []

    for _ in range(max_len):
        dec_embedded = decoder_embedding(decoder_input)
        context, _ = attention(decoder_hidden[-1], encoder_outputs, src_mask)
        decoder_input_lstm = torch.cat([dec_embedded, context], dim=1).unsqueeze(1)

        decoder_output, (decoder_hidden, decoder_cell) = decoder(
            decoder_input_lstm, (decoder_hidden, decoder_cell)
        )

        prediction = output_layer(decoder_output.squeeze(1))
        next_id = prediction.argmax(dim=-1).item()

        if next_id == EOS:
            break

        generated_ids.append(next_id)
        decoder_input = torch.tensor([next_id], dtype=torch.long).to(device)

    return sp.decode(generated_ids)


# ---------------------------------------------------------------------
# Beam search decoding
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_question_beam(src_text, beam_width=3, max_len=MAX_LEN, length_penalty=0.7):
    encoder_outputs, src_mask, init_hidden, init_cell = _encode_source(src_text)

    beams = [([BOS], 0.0, init_hidden, init_cell, False)]

    for _ in range(max_len):
        candidates = []

        for tokens, score, d_hidden, d_cell, finished in beams:
            if finished:
                candidates.append((tokens, score, d_hidden, d_cell, finished))
                continue

            decoder_input = torch.tensor([tokens[-1]], dtype=torch.long).to(device)
            dec_embedded = decoder_embedding(decoder_input)
            context, _ = attention(d_hidden[-1], encoder_outputs, src_mask)
            decoder_input_lstm = torch.cat([dec_embedded, context], dim=1).unsqueeze(1)

            decoder_output, (new_hidden, new_cell) = decoder(
                decoder_input_lstm, (d_hidden, d_cell)
            )

            logits = output_layer(decoder_output.squeeze(1))
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for lp, idx in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                candidates.append((
                    tokens + [idx],
                    score + lp,
                    new_hidden,
                    new_cell,
                    idx == EOS
                ))

        candidates.sort(
            key=lambda c: c[1] / (len(c[0]) ** length_penalty),
            reverse=True
        )
        beams = candidates[:beam_width]

        if all(b[4] for b in beams):
            break

    best_tokens = beams[0][0]
    generated_ids = [t for t in best_tokens if t not in (BOS, EOS)]
    return sp.decode(generated_ids)


# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    sentence = ""
    answer = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        answer = request.form.get("answer", "").strip()

        if not sentence or not answer:
            error = "Please provide both the sentence and the answer span."
        elif answer not in sentence:
            error = "The answer text was not found inside the sentence. Check spelling/spacing."
        else:
            # Wrap the answer span in <ans> ... </ans>, same as training preprocessing
            idx = sentence.find(answer)
            marked_source = (
                sentence[:idx] + f"{ANS_OPEN} {answer} {ANS_CLOSE}" + sentence[idx + len(answer):]
            )
            marked_source = " ".join(marked_source.split())

            greedy_output = generate_question(marked_source)
            beam_output = generate_question_beam(marked_source, beam_width=3)

            result = {
                "marked_source": marked_source,
                "greedy": greedy_output,
                "beam": beam_output,
            }

    return render_template(
        "index.html",
        result=result,
        error=error,
        sentence=sentence,
        answer=answer,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
