import os
import sys
import numpy as np

import torch
import utils

from scipy.io import wavfile
from text.symbols import symbols
from text import cleaned_text_to_sequence, sequence_to_text
from vits_pinyin import VITS_PinYin
import IPython.display as ipd
from tqdm import tqdm
import re
import random
import soundfile as sf
import json
from pydub import AudioSegment
import gc


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

replace_character = [
    "嗯", "额", "哼", "啊", "就是", "然后", "这个", "那个", "其实"
]

frequency = 0.016

def get_hanzi(str):
    pattern = r'[\u4e00-\u9fa5]'
    hanzi_list = re.findall(pattern, str)
    result = ''.join(hanzi_list)
    return result


def insert_sil(audio, sr, insert_point_sec, duration_sec):
    insert_point_samples = int(insert_point_sec * sr)
    silence_length_samples = int(duration_sec * sr)
    
    silence = np.zeros(silence_length_samples)
    
    audio_first_part = audio[:insert_point_samples]
    audio_second_part = audio[insert_point_samples:]
    new_audio = np.concatenate((audio_first_part, silence, audio_second_part))
    
    return new_audio




def infer_audio(text, sid, tts_front, net_g):
    phonemes, char_embeds = tts_front.chinese_to_phonemes(text)

    # char_embeds = new_embeds
    input_ids = cleaned_text_to_sequence(phonemes)  ### text encoder
    # print("phonemes:" + str(phonemes))
    # print("input_ids", input_ids)
    # print("length: ", len(input_ids))
    # print("char_embeds: " + str(char_embeds.shape)) # [L, 256]
    
    with torch.no_grad():
        sid = torch.LongTensor([12]).to(device)
        x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        # print("x_tst: " + str(x_tst))
        x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        # print("x_tst_prosody: " + str(x_tst_prosody))
        output = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, sid=sid, noise_scale=0.5, length_scale=1)
        audio = output[0][0, 0].data.cpu().float().numpy()
        w_ceil = output[-1]
    
    return audio, w_ceil, phonemes


def infer_prolong_audio(text, sid, tts_front, net_g):
    phonemes, char_embeds = tts_front.chinese_to_phonemes(text)

    # char_embeds = new_embeds
    input_ids = cleaned_text_to_sequence(phonemes)  ### text encoder
    # print("phonemes:" + str(phonemes))
    # print("input_ids", input_ids)
    # print("length: ", len(input_ids))
    # print("char_embeds: " + str(char_embeds.shape)) # [L, 256]
    
    with torch.no_grad():
        sid = torch.LongTensor([12]).to(device)
        x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        # print("x_tst: " + str(x_tst))
        x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        # print("x_tst_prosody: " + str(x_tst_prosody))
        output = net_g.infer_prolong(x_tst, x_tst_lengths, x_tst_prosody, sid=sid, noise_scale=0.5, length_scale=1)
        audio = output[0][0, 0].data.cpu().float().numpy()
        index = output[-2]
        w_ceil = output[-1]

    return audio, w_ceil, index, phonemes


def get_time_transcription(w_ceil, phonemes):
    dur = w_ceil * frequency
    phonemes = phonemes.split(' ')

    durations = dur.squeeze()
    cumulative_sums = durations.cumsum(0)
    start_indices = torch.zeros_like(cumulative_sums)
    start_indices[1:] = cumulative_sums[:-1]
    start_indices[0] = 0
    end_indices = cumulative_sums
    character_starts = start_indices[::2]
    character_starts = character_starts[1:]
    character_ends = end_indices[1::2]  
    character_ends = character_ends[1:len(character_starts)]

    ret = []
    for i, (start, end) in enumerate(zip(character_starts, character_ends)):
        phoneme = phonemes[2*i + 1] + phonemes[2*i + 2]
        ret.append({
            "phoneme": phoneme,
            "start": round(start.item(), 3),  
            "end": round(end.item(), 3),   
            "type": None
        })
    
    return ret


####################################
def generate_rep(text, sid, tts_front, net_g, out_path): # text: hazi str
    length = len(text)
    rep_index = random.randint(0, length - 1) # [0, len]
    # print('rep_index: ', rep_index)
    rep_num = random.randint(3, 5)
    
    rep_text = text[:rep_index] + text[rep_index] * rep_num + text[rep_index + 1:]

    audio, w_ceil, x = infer_audio(rep_text, sid, tts_front, net_g)
    sf.write(out_path, audio, samplerate=16000)

    timestamps = get_time_transcription(w_ceil, x)
    # print(timestamps)

    merged_start = timestamps[rep_index]['start']
    merged_end = timestamps[rep_index + rep_num -1]['end']
    phoneme = timestamps[rep_index]['phoneme']

    merged_element = {'phoneme': phoneme, 'start': merged_start, 'end': merged_end, 'type': "rep"}
    timestamps[rep_index : rep_index + rep_num] = [merged_element]
    # print(timestamps)

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]

    json_path = out_path.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)



def generate_miss(text, sid, tts_front, net_g, out_path):
    length = len(text)
    miss_index = random.randint(0, length - 1) # [0, len]

    miss_text = text[:miss_index] + text[miss_index + 1:]

    audio, w_ceil, x = infer_audio(miss_text, sid, tts_front, net_g)
    sf.write(out_path, audio, samplerate=16000)

    timestamps = get_time_transcription(w_ceil, x)
    # print(timestamps)

    miss_time = timestamps[miss_index - 1]['end']

    timestamps.insert(miss_index, {
                "phoneme": text[miss_index],
                "start": miss_time,
                "end": miss_time,
                "type": "missing"
            })
    # print(timestamps)

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]

    json_path = out_path.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)



def generate_replace(text, sid, tts_front, net_g, out_path):
    length = len(text)
    replace_index = random.randint(0, length - 1) # [0, len]

    replace_ch = random.choice(replace_character) 
    # print(replace_ch)

    replace_text = text[:replace_index] + replace_ch + text[replace_index + 1:]
    # print(replace_text)

    audio, w_ceil, x = infer_audio(replace_text, sid, tts_front, net_g)
    sf.write(out_path, audio, samplerate=16000)

    timestamps = get_time_transcription(w_ceil, x)
    # print(timestamps)

    if len(replace_ch) == 1:
        timestamps[replace_index]['type'] = "replace"
    else:
        merged_start = timestamps[replace_index]['start']
        merged_end = timestamps[replace_index + 1]['end']
        phoneme = timestamps[replace_index]['phoneme'] + timestamps[replace_index + 1]['phoneme']

        merged_element = {'phoneme': phoneme, 'start': merged_start, 'end': merged_end, 'type': "replace"}
        timestamps[replace_index : replace_index + 2] = [merged_element]
    
    # print(timestamps)

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]

    json_path = out_path.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)



def generate_prolong(text, sid, tts_front, net_g, out_path):
    audio, w_ceil, index, x = infer_prolong_audio(text, sid, tts_front, net_g)
    # print("w_ceil: ", w_ceil)
    # print("index: ", index)
    sf.write(out_path, audio, samplerate=16000)

    timestamps = get_time_transcription(w_ceil, x)
    # print(timestamps)

    prolong_index = (index - 1)//2 - 1
    timestamps[prolong_index]['type'] = "prolong"

    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]

    json_path = out_path.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)



def generate_block(text, sid, tts_front, net_g, out_path):
    audio, w_ceil, x = infer_audio(text, sid, tts_front, net_g)
    # print("audio: ", audio) 
    timestamps = get_time_transcription(w_ceil, x)
    # print(timestamps)

    block_index = random.randint(0, len(text) - 1)
    chosen = timestamps[block_index]
    # print(chosen)
    silence_duration = random.randint(1,4)
    # print("sil: ", silence_duration)
    
    audio = insert_sil(audio, 16000, chosen["end"], silence_duration)
    sf.write(out_path, audio, samplerate=16000)
    
    block_phoneme = {
        "phoneme": None,
        "start": chosen["end"],
        "end": round(chosen["end"] + silence_duration, 3),
        "type": "block"
    }
    block_phoneme
    timestamps.insert(block_index + 1, block_phoneme)

    for i in range(block_index + 2, len(timestamps)):
        timestamps[i]["start"] += silence_duration
        timestamps[i]["end"] += silence_duration
        timestamps[i]["start"] = round(timestamps[i]["start"], 3)
        timestamps[i]["end"] = round(timestamps[i]["end"], 3)


    label = [{
        "start": round(timestamps[0]["start"], 3),
        "end": round(timestamps[-1]["end"], 3),
        "phonemes": timestamps
    }]
    
    json_path = out_path.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)




def add_dysfluency(text, sid, tts_front, net_g, out_path):
    generate_rep(text, sid, tts_front, net_g, out_path.replace('.wav', '_rep.wav'))
    generate_miss(text, sid, tts_front, net_g, out_path.replace('.wav', '_missing.wav'))
    generate_replace(text, sid, tts_front, net_g, out_path.replace('.wav', '_replace.wav'))
    generate_prolong(text, sid, tts_front, net_g, out_path.replace('.wav', '_prolong.wav'))
    generate_block(text, sid, tts_front, net_g, out_path.replace('.wav', '_block.wav'))




if __name__ == '__main__':
    # pinyin
    tts_front = VITS_PinYin("./bert", device)

    # config
    hps = utils.get_hparams_from_file("configs/bert_vits.json")

    # model
    net_g = utils.load_class(hps.train.eval_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)

    utils.load_model("AISHELL3_G.pth", net_g)
    net_g.eval()
    net_g.to(device)

    txt_file = "AISHELL3.txt"
    for speaker_id in tqdm(range(0, 174), desc="processing speakers"):
        sid = torch.LongTensor([speaker_id]).cuda()  # Set speaker id

        with open(txt_file, 'r', encoding='utf-8') as file:
            index = 0
            for line in tqdm(file, total=420, desc=f"Generating for Speaker {speaker_id}", leave=False):
                line = line.strip()
                sid_value = sid.item()
                filename = f"p{sid_value:03}_{index:03}.wav"

                out_path = f"/home/xuanru/tts/vits_chinese/output/disfluent_audio/{filename}"
                text_file_path = f"/home/xuanru/tts/vits_chinese/output/gt_text/{filename.replace('.wav', '.txt')}"
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(line) 
                try:
                    add_dysfluency(line, sid, tts_front, net_g, out_path)
                    index += 1
                except Exception as e:
                    print("error:{}".format(out_path))
                    continue
            
        torch.cuda.empty_cache()
        gc.collect()