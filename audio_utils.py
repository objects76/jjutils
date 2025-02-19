
#%% - m3u
from typing import cast, Iterable
import sys, os, re
if os.path.abspath('.') not in sys.path: sys.path.append(os.path.abspath('.'))

# import os, huggingface_hub # !pip install huggingface_hub[hf_transfer]
# huggingface_hub.login(token = os.environ.get('HF_TOKEN'), add_to_git_credential=True)

from pathlib import Path
from pprint import pprint
from glob import glob
import numpy as np; np.set_printoptions(precision=8, suppress=True); np.random.seed(42)
# import pandas as pd
# from moviepy.editor import VideoFileClip
import subprocess

#%% mp4 info
# def get_video_duration(file_path):
#     with VideoFileClip(file_path) as video:
#         return video.duration  # duration is in seconds

# if __name__ == '__main__':
#     mp4file = 'testdata/jp.Meeting.mp4'
#     duration = get_video_duration(mp4file)
#     print(f"The video duration is {duration} seconds.")

#%% - m3u
def time_to_seconds(time_str):
    return hhmmss_to_seconds(time_str)


def hhmmss_to_seconds(time_str) -> float:
    if not time_str: return 0
    time_str = time_str.strip()
    if time_str.count(':') == 1: time_str = "00:" + time_str
    millis = 0
    hours, minutes, seconds = time_str.split(':')
    if ',' in seconds:
      seconds, millis = seconds.split(',')
    elif '.' in seconds:
      seconds, millis = seconds.split('.')

    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0
    return total_seconds

def to_hhmmss(seconds, compact=False):
    secs, _, frac = str(seconds).partition('.')
    ms = (frac + '000')[:3]
    secs = int(secs)
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    seconds = secs % 60

    if compact:
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"

    return f"{hours:02}:{minutes:02}:{seconds:02},{ms}"


FFMPEG = 'ffmpeg -nostats -hide_banner -y '

# import pydub
# sound = pydub.AudioSegment.from_file('testdata/jp.20240319.mp4').set_channels(1)
# sound.export("tmp/jp.20240319.wav", format="wav", codec='pcm_s16le', bitrate='128k', parameters="-ar 16000".split())

def get_audio(mp4file, *, outdir='./tmp', start_time = 0, end_time = 0, audio_type = 'mp3') ->str:
    """
    get audio file from mp4 file
    """
    if isinstance(start_time, int) and isinstance(end_time, int):
        ssec = start_time
        esec = end_time
    else:
        ssec, esec = time_to_seconds(start_time), time_to_seconds(end_time)

    if esec - ssec <= 0:
        audio_file = mp4file.replace('.mp4', f'.{audio_type}')
    else:
        audio_file = mp4file.replace('.mp4', f'-{ssec}_{esec}.{audio_type}')
        assert False, 'deprecated with start, end'

    audio_file = Path(mp4file).with_suffix('.'+audio_type)

    audio_file = Path(outdir) / Path(audio_file).name

    if not Path(audio_file).exists() or False:
        Path(outdir).mkdir(exist_ok=True)
        if audio_type == 'wav':
            # cmd = f"ffmpeg -nostdin -threads 0 -i {audiofile} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
            # !ffmpeg -i testdata/ntt.meeting.mp4-0_0.mp3 -ac 1 -ar 16000 testdata/ntt.meeting_16k.wav
            ac = '-ac 1'
            ar = '-ar 16000'
            if ssec == 0 and esec == 0:
                cmds = f'{FFMPEG} -i {mp4file} -vn -acodec pcm_s16le {ac} {ar} {audio_file}'
            else:
                cmds = f'{FFMPEG} -ss {ssec} -to {esec} -i {mp4file} -vn -acodec pcm_s16le {ac} {ar} {audio_file}'
        elif audio_type == 'mp3':
            if ssec == 0 and esec == 0:
                cmds = f'{FFMPEG} -i {mp4file} -vn -acodec mp3 -b:a 192k {audio_file}'
            else:
                cmds = f'{FFMPEG} -ss {ssec} -to {esec} -i {mp4file} -vn -acodec mp3 -b:a 192k {audio_file}'

        print(cmds)
        subprocess.run(cmds.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return audio_file.as_posix()

def replace_mp4_audio(mp4file, audioin, mp4output):
    cmds = f'{FFMPEG} -i {mp4file} -i {audioin} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {mp4output}'
    # print(cmds)
    subprocess.run(cmds.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def audio2mp4(wav_path, out_path):
    assert Path(wav_path).exists()
    if Path(out_path).exists(): return
    jpg_path = 'testdata/1280x720-bg.jpg'
    FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
    cmd = f"{FFMPEG} -i {wav_path} -loop 1 -i {jpg_path} -c:v libx264 -c:a aac -b:a 192k -shortest {out_path}"
    print(cmd)
    subprocess.run(cmd.split(), check=True)




import re
def get_segment_from_srt(from_srt:Path):
    if from_srt:
        with open(from_srt) as fp: srt = fp.read()
        matches = re.findall(r"([\d:,]+) --> ([\d:,]+)\n([^\n]+)", srt, re.M)
        segfile_path = str(from_srt).replace('.srt', '.seg')
        with open(segfile_path, 'w') as fp:
            fp.write('start\tend\text\n')
            for m in matches:
                s,e,t = m
                fp.write(f"{s}\t{e}\t{t}\n")

from pydub import AudioSegment, generators # pip install pydub
import simpleaudio # sudo apt-get install libasound2-dev && pip install simpleaudio
from pyannote.core import Segment
import time

from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play

def play_beep(frequency=440, dur_sec=0.5):
    beep = Sine(frequency).to_audio_segment(duration=int(dur_sec*1000))
    play(beep)


def play_audio(file_path: str,
               *,
            #    ranges: list[ tuple[float,float] ]| list[Segment],
               ranges: Iterable,
               speed: float = 1.0,
               start_end_notifier = True):
    # Load the full audio file
    audio = AudioSegment.from_file(file_path)

    if start_end_notifier:
        beep_duration = 100  # duration in milliseconds
        beep_frequency = 550  # frequency in Hz
        beep_volume = -20  # reduce the beep volume in dB
        beep_start = generators.Sine(450).to_audio_segment(duration=beep_duration).apply_gain(beep_volume)
        beep_end = generators.Sine(600).to_audio_segment(duration=beep_duration).apply_gain(beep_volume)
        mute = AudioSegment.silent(duration=800)


    segments = []
    for (start, end) in ranges:
        # Extract the specific range (times are in milliseconds)
        segment = audio[start * 1000:end * 1000]
        if start_end_notifier:
            segment = beep_start + segment + beep_end + mute

        # Speed up the segment
        if speed != 1.0:
            # Change the frame rate to speed up or slow down the playback
            new_frame_rate = int(segment.frame_rate * speed)
            segment = segment.set_frame_rate(new_frame_rate)

        segments.append(segment)

    from IPython.display import display, update_display, DisplayHandle
    handle = cast(DisplayHandle, display('', display_id=True))

    # Play each segment
    n_total = len(segments)
    for i, (segment, (start,end)) in enumerate(zip(segments, ranges), start=1):
        # Convert segment to raw audio data for playback
        # print(f'{start:.1f} ~ {end.1f}: {end-start:.1f}sec')
        raw_data = segment.raw_data
        wave_obj = simpleaudio.WaveObject(raw_data, num_channels=segment.channels,
                                 bytes_per_sample=segment.sample_width, sample_rate=segment.frame_rate)
        play_obj = wave_obj.play()
        update_display(f'> {start:.3f} ~ {end:.3f}({end-start:.3f}), {i}/{n_total}', display_id=handle.display_id)
        if len(ranges) == 1:
            return play_obj

        while play_obj.is_playing():
            time.sleep(0.05)

def stop_plays():
    simpleaudio.stop_all()

#%%
import subprocess
import json

def get_audio_info(file_path):
    if not file_path or not Path(file_path).exists():
        return
    # https://ottverse.com/ffprobe-comprehensive-tutorial-with-examples/
    command = 'ffprobe -v error -show_entries format=duration:stream=bit_rate,sample_rate,channels -of json ' + file_path
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Convert output from JSON string to Python dictionary
    try:
        info = json.loads(result.stdout)
        return info
    except (KeyError, json.JSONDecodeError) as e:
        print("Error parsing information:", e)
        return None

#%%
from pydub import AudioSegment, generators
import simpleaudio as sa # sudo apt-get install libasound2-dev && pip install simpleaudio

def get_audio_segments(
        file_path: Path,
        *,
        pklsegs: list,
        start_end_notifier = True,
        ) -> list[AudioSegment]:
    # Load the full audio file
    assert file_path.suffix == '.wav' or file_path.suffix == '.mp3'
    audio = AudioSegment.from_file(file_path)

    # Generate a 0.1-second beep at 550 Hz (A4 note)
    beep_duration = 100  # duration in milliseconds
    beep_frequency = 550  # frequency in Hz
    beep_volume = -20  # reduce the beep volume in dB
    beep_start = generators.Sine(450).to_audio_segment(duration=beep_duration).apply_gain(beep_volume)
    beep_end = generators.Sine(600).to_audio_segment(duration=beep_duration).apply_gain(beep_volume)

    wave_segments = []
    for seg in pklsegs:

        # Extract the specific range (times are in milliseconds)
        segment = audio[seg.start_sec * 1000:seg.end_sec * 1000]

        if start_end_notifier:
            segment = beep_start + segment + beep_end

        wave_segments.append(segment)
    return wave_segments


def play_segment(segment:AudioSegment, speed:float=1.0):
    if speed != 1.0:
        new_frame_rate = int(segment.frame_rate * speed)
        segment = segment.set_frame_rate(new_frame_rate)

    raw_data = segment.raw_data
    wave_obj = sa.WaveObject(raw_data, num_channels=segment.channels,
                            bytes_per_sample=segment.sample_width, sample_rate=segment.frame_rate)
    play_obj = wave_obj.play()
    return play_obj

# trim audio with ffmpeg
#    ffmpeg -i testdata/jp.zoom-4person.mp4 -ss 00:01:00 -t 00:30:00 -c copy testdata/jp.zoom-4person-trimmed.mp4
#    ffmpeg -i testdata/jp.zoom-4person.mp4 -ss 00:31:00  -c copy testdata/jp.zoom-4person-trimmed2.mp4


import numpy as np
import subprocess
import torch

def load_audio(media_input: str|Path, normalize=True) -> torch.Tensor:
    FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
    try:
        cmd = f"{FFMPEG} -i {str(media_input)} -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -"
        out:bytes = subprocess.run(cmd.split(), capture_output=True, check=True).stdout
        waveform = np.frombuffer(out, np.int16).flatten() # int16
        if normalize:
            waveform = waveform.astype(np.float32) / 32768.0 # float32

        return torch.from_numpy(waveform[None, :]) # [1,n_signal]

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


import torch
import numpy as np
import simpleaudio
from pyannote.core import Segment, Timeline, Annotation
from typing import Final


def play_tensor(audio_tensor:torch.Tensor, play_sec:int=999, sr=16000):
    if audio_tensor.dtype == torch.int16:
        audio_tensor = audio_tensor.float() / 32768.0
        print(audio_tensor.shape, audio_tensor.dtype)

    audio_data = audio_tensor.squeeze().numpy()
    assert audio_data.dtype == np.float32, f"np.float32? {audio_data.dtype}"
    assert audio_data.ndim == 1, f"ndim==1? {audio_data.ndim=} {audio_data.shape=}"
    audio_data = audio_data[:16000*play_sec]
    assert -1 <= torch.min(audio_tensor) and torch.max(audio_tensor) <= 1, f"[-1,1] =? {torch.min(audio_tensor)} ~ {torch.max(audio_tensor)}"
    audio_data = (audio_data * 32767).astype(np.int16) # normalized float -> int16
    try:
        audio_bytes = audio_data.tobytes()
        play_obj = simpleaudio.play_buffer(audio_bytes, num_channels=1, bytes_per_sample=2, sample_rate=sr)
        play_obj.wait_done()
    except KeyboardInterrupt:
        play_obj.stop()
        print("KeyboardInterrupt")
        pass


def get_audio_clips(
        segments:Timeline,
        mp4path:str,
        min_threshold_sec:float=0
        ) -> list[torch.Tensor]:
    """
    return list[ Tensor(1,n_signal) ]
    """
    self = get_audio_clips
    if self._audio_data.get("mp4path", "") != mp4path:
        self._audio_data["pcm16k"] = load_audio(mp4path)
        self._audio_data["mp4path"] = mp4path

    pcm16k:torch.Tensor = self._audio_data["pcm16k"]
    min_signal:Final[int] = int(min_threshold_sec *16000)

    waves = []
    for seg in iter(segments):
        chunk = pcm16k[:, int(seg.start*16000):int(seg.end*16000)]

        if len(chunk[-1]) < min_signal:
            # print(f"{len(chunk[-1])/16000:.3f} sec dropped, {chunk.shape=}")
            continue
        waves.append(chunk)
    print(f"{len(waves)} / {len(segments)}, #{len(segments)-len(waves)} dropped")
    return waves
get_audio_clips._audio_data = {}


import subprocess
def audio_to_mp4(audio_path, out_path):
    assert Path(audio_path).exists()
    if Path(out_path).exists():
        print(out_path, "already exists")
        return
    jpg_path = 'testdata/1280x720-bg.jpg'
    FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
    cmd = f"{FFMPEG} -i {audio_path} -loop 1 -i {jpg_path} -c:v libx264 -c:a aac -b:a 192k -shortest {out_path}"
    print(cmd)
    subprocess.run(cmd.split(), check=True)
#
# audio_to_mp4(
#     "testdata/ko/rsupkr-6c604986-cf3f-40ee-acd3-f29a3595ebe0.webm",
#     "testdata/ko/rsupkr-6c604986-cf3f-40ee-acd3-f29a3595ebe0.mp4"
#         )


#
# if __name__ == '__main__':
#     mp4file = 'testdata/ntt.meeting.mp4'
#     get_audio(mp4file, start_time=720, end_time=1500, audio_type='wav')


