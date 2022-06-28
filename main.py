import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")


audio_waveform, sample_rate = torchaudio.load('./audio.wav')

inputs = processor(audio_waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values.to("cpu")).logits
pred_ids = torch.argmax(logits, dim=-1)

scores = torch.nn.functional.log_softmax(logits, dim=-1)

output = processor.batch_decode(pred_ids, output_word_offsets=False, output_char_offsets=False)

print(output)
