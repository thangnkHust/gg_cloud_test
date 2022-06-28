from pyannote.core import Segment, notebook
from pyannote.audio import Audio
from pyannote.audio import Pipeline
import torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from IPython.display import Audio as IPythonAudio
import math
import time

input_file = './_5K9_3Wi6i.wav'
audio = Audio(sample_rate=16000, mono=True)
duration = audio.get_duration(input_file)
time_per_split = 10
temp = math.ceil(duration/time_per_split)

# init pipeline for speaker diarization
pipeline = Pipeline.from_pretrained('speaker_diarization/config.yaml')
# initial_params = {
#   "min_duration_on": 0.0, "min_duration_off": 1.0
# }
# pipeline.instantiate(initial_params)

def speaker_diarization(input_file, start_time, end_time):
    excerpt = Segment(start=start_time, end=end_time)
    waveform, sample_rate = audio.crop(input_file, excerpt)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    return diarization.for_json()['content']

# fake speaker identification
def speaker_identification(input_file, start_time, end_time, speaker_name_fake):
  excerpt = Segment(start=start_time, end=end_time)
  waveform, sample_rate = audio.crop(input_file, excerpt)
  switcher = {
      'SPEAKER_00': {
          'name': 'A',
          'confidence': 0.8
      },
      'SPEAKER_01': {
          'name': 'B',
          'confidence': 0.7
      },
      'SPEAKER_02': {
          'name': 'C',
          'confidence': 0.85
      },
      'SPEAKER_03': {
          'name': 'D',
          'confidence': 0.8
      },
      'SPEAKER_04': {
          'name': 'E',
          'confidence': 0.8
      },
      'SPEAKER_05': {
          'name': 'F',
          'confidence': 0.8
      }
  }
  return switcher.get(speaker_name_fake, {'name': 'unknown', 'confidence': None})

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

from dataclasses import dataclass
def get_trellis(emission, tokens, blank_id=0):
  num_frame = emission.size(0)
  num_tokens = len(tokens)

  # Trellis has extra diemsions for both time axis and tokens.
  # The extra dim for tokens represents <SoS> (start-of-sentence)
  # The extra dim for time axis is for simplification of the code.
  trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
  trellis[:, 0] = 0
  for t in range(num_frame):
    trellis[t+1, 1:] = torch.maximum(
        # Score for staying at the same token
        trellis[t, 1:] + emission[t, blank_id],
        # Score for changing to the next token
        trellis[t, :-1] + emission[t, tokens],
    )
  return trellis

@dataclass
class Point:
  token_index: int
  time_index: int
  score: float


def backtrack(trellis, emission, tokens, blank_id=0):
  # Note:
  # j and t are indices for trellis, which has extra dimensions
  # for time and tokens at the beginning.
  # When refering to time frame index `T` in trellis,
  # the corresponding index in emission is `T-1`.
  # Similarly, when refering to token index `J` in trellis,
  # the corresponding index in transcript is `J-1`.
  j = trellis.size(1) - 1
  t_start = torch.argmax(trellis[:, j]).item()

  path = []
  for t in range(t_start, 0, -1):
    # 1. Figure out if the current position was stay or change
    # Note (again):
    # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
    # Score for token staying the same from time frame J-1 to T.
    stayed = trellis[t-1, j] + emission[t-1, blank_id]
    # Score for token changing from C-1 at T-1 to J at T.
    changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

    # 2. Store the path with frame-wise probability.
    prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
    # Return token index and time index in non-trellis coordinate.
    path.append(Point(j-1, t-1, prob))

    # 3. Update the token
    if changed > stayed:
      j -= 1
      if j == 0:
        break
  else:
    raise ValueError('Failed to align')
  return path[::-1]

@dataclass
class SegmentTest:
  label: str
  start: int
  end: int
  score: float
  start_time: float = None
  end_time: float = None

  def __repr__(self):
    return f"{self.label}\t({self.score:4.4f}): [{self.start:5d}, {self.end:5d})\t[{self.start_time:4.4f}, {self.end_time:4.4f})"

  @property
  def length(self):
    return self.end - self.start

def merge_repeats(path, output):
  segments = []
  for item in output.char_offsets[0]:
    i1 = item['start_offset']
    i2 = item['end_offset']
    score = sum(point.score for point in path[i1:i2]) / (i2 - i1)
    segments.append(SegmentTest(item['char'], path[i1].time_index, path[i2-1].time_index + 1, score))
  return segments

def merge_words(segments, time_per_offset, separator=' '):
  words = []
  i1, i2 = 0, 0
  while i1 < len(segments):
    if i2 >= len(segments) or segments[i2].label == separator:
      if i1 != i2:
        segs = segments[i1:i2]
        word = ''.join([seg.label for seg in segs])
        score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
        words.append(SegmentTest(word, segments[i1].start, segments[i2-1].end, score, segments[i1].start*time_per_offset, segments[i2-1].end*time_per_offset))
      i1 = i2 + 1
      i2 = i1
    else:
      i2 += 1
  return words

def speech_to_text(input_file, start_time, end_time):
  excerpt = Segment(start=start_time, end=end_time)
  waveform, sample_rate = audio.crop(input_file, excerpt)
  inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
  with torch.no_grad():
    logits = model(inputs.input_values.to("cpu")).logits
  pred_ids = torch.argmax(logits, dim=-1)
  scores = torch.nn.functional.log_softmax(logits, dim=-1)

  output = processor.batch_decode(pred_ids, output_word_offsets=True, output_char_offsets=True)

  emission = scores[0].cpu().detach()
  dictionary  = processor.tokenizer.get_vocab()
  tokens = pred_ids.cpu().detach().squeeze().numpy()

  trellis = get_trellis(emission, tokens)
  path = backtrack(trellis, emission, tokens)

  segments = merge_repeats(path, output)
  time_per_offset = (end_time - start_time)/len(pred_ids[0])
  word_segments = merge_words(segments, time_per_offset)
  return word_segments

# Main run SD + SI + S2T
result_temp = []
for i in range(temp):
  print(f'handle {i}')
  start_time = i*time_per_split
  if i == temp-1:
    end_time = duration
  else:
    end_time = (i+1)*time_per_split
  result_diarization = speaker_diarization(input_file, start_time, end_time)

  for spk in result_diarization:
    spk['segment']['start'] += start_time
    spk['segment']['end'] += start_time
    spk['label'] = speaker_identification(input_file, spk['segment']['start'], spk['segment']['end'], spk['label'])
  result_temp.append(result_diarization)

result = []
k = 0
for i in range(len(result_temp)):
  for j in range(k, len(result_temp[i])):
    if i != len(result_temp) - 1:
      if j == len(result_temp[i]) - 1 and result_temp[i][j]['label']['name'] == result_temp[i+1][0]['label']['name']:
        result_temp[i][j]['segment']['end'] = result_temp[i+1][0]['segment']['end']
        result_temp[i][j]['text'] = speech_to_text(input_file, result_temp[i][j]['segment']['start'], result_temp[i][j]['segment']['end'])
        result.append(result_temp[i][j])
        k = 1
      else:
        result_temp[i][j]['text'] = speech_to_text(input_file, result_temp[i][j]['segment']['start'], result_temp[i][j]['segment']['end'])
        result.append(result_temp[i][j])
        k = 0
    else:
      result_temp[i][j]['text'] = speech_to_text(input_file, result_temp[i][j]['segment']['start'], result_temp[i][j]['segment']['end'])
      result.append(result_temp[i][j])
      k = 0


print(result)
