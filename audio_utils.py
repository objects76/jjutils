
#%% - m3u
from typing import cast, Iterable
import sys, os, re

from dbg import is_interactive
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

def get_audio(mp4file, *, outdir='./tmp', start_time = 0., end_time = 0., audio_type = 'mp3') ->str:
    """
    get audio file from mp4 file
    """
    ssec = start_time
    esec = end_time
    if isinstance(start_time, str) and isinstance(end_time, str):
        ssec, esec = time_to_seconds(start_time), time_to_seconds(end_time)

    ext = audio_type if audio_type[0] == '.' else '.'+audio_type
    if esec - ssec <= 0:
        output_path = mp4file.replace('.mp4', ext)
    else:
        output_path = mp4file.replace('.mp4', f'-{ssec}_{esec}{ext}')
        # assert False, 'deprecated with start, end'

    output_path = Path(outdir) / Path(output_path).name
    if not output_path.exists() or False:
        Path(outdir).mkdir(exist_ok=True)
        if ext == '.wav':
            # cmd = f"ffmpeg -nostdin -threads 0 -i {audiofile} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
            # !ffmpeg -i testdata/ntt.meeting.mp4-0_0.mp3 -ac 1 -ar 16000 testdata/ntt.meeting_16k.wav
            ac = '-ac 1'
            ar = '-ar 16000'
            if ssec == 0 and esec == 0:
                cmds = f'{FFMPEG} -i {mp4file} -vn -acodec pcm_s16le {ac} {ar} {output_path}'
            else:
                cmds = f'{FFMPEG} -ss {ssec} -to {esec} -i {mp4file} -vn -acodec pcm_s16le {ac} {ar} {output_path}'
        elif ext == '.mp3':
            if ssec == 0 and esec == 0:
                cmds = f'{FFMPEG} -i {mp4file} -vn -acodec mp3 -b:a 192k {output_path}'
            else:
                cmds = f'{FFMPEG} -ss {ssec} -to {esec} -i {mp4file} -vn -acodec mp3 -b:a 192k {output_path}'

        print(cmds)
        subprocess.run(cmds.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_path.as_posix()

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

def play_beep(frequency=440, dur_sec=0.5, sleep_sec=0):
    beep = Sine(frequency).to_audio_segment(duration=int(dur_sec*1000))
    play(beep)
    if sleep_sec>0:
        time.sleep(sleep_sec)

def play_audio(file_path: str,
               *,
               ranges: list[Segment],
               speed: float = 1.0,
               start_end_notifier = True,
               mute_sec = 2,
               output_m4a:str = ""):
    if len(ranges) == 0: return
    # Load the full audio file and cache the audio inst.
    if play_audio.file_path != file_path:
        play_audio.file_path = file_path
        play_audio.audio = AudioSegment.from_file(file_path)
    audio = play_audio.audio

    if start_end_notifier:
        beep_duration = 100  # duration in milliseconds
        beep_frequency = 550  # frequency in Hz
        beep_volume = -20  # reduce the beep volume in dB
        beep_start = generators.Sine(450).to_audio_segment(duration=beep_duration).apply_gain(beep_volume)
        beep_end = generators.Sine(600).to_audio_segment(duration=beep_duration).apply_gain(beep_volume)
        mute = AudioSegment.silent(duration=int(mute_sec*1000))


    segments = []
    for (start, end) in ranges:
        # Extract the specific range (times are in milliseconds)
        segment = audio[int(start * 1000):int(end * 1000)]
        if start_end_notifier:
            segment = beep_start + segment + beep_end + mute
            # segment = segment + beep_end + mute

        # Speed up the segment
        if speed != 1.0:
            # Change the frame rate to speed up or slow down the playback
            new_frame_rate = int(segment.frame_rate * speed)
            segment = segment.set_frame_rate(new_frame_rate)

        segments.append(segment)

    # Export each segment to an audio file
    if output_m4a.endswith('.m4a'):
        combined_audio = AudioSegment.silent(duration=0)  # Start with a silent audio segment
        for i, (segment, (start, end)) in enumerate(zip(segments, ranges), start=1):
            # Append the segment to the combined audio
            combined_audio += segment

        # Export the combined audio to a file
        combined_audio.export(output_m4a, format="ipod")  # Use 'ipod' for m4a format
        print(f"Exported combined audio to {output_m4a}")


    n_total = len(segments)
    from IPython.display import display, update_display, DisplayHandle
    handle = cast(DisplayHandle, display('', display_id=True)) if n_total>=3 else None

    # Play each segment
    for i, (segment, (start,end)) in enumerate(zip(segments, ranges), start=1):
        # Convert segment to raw audio data for playback
        # print(f'{start:.1f} ~ {end.1f}: {end-start:.1f}sec')
        raw_data = segment.raw_data
        wave_obj = simpleaudio.WaveObject(raw_data, num_channels=segment.channels,
                                 bytes_per_sample=segment.sample_width, sample_rate=segment.frame_rate)
        play_obj = wave_obj.play()
        if handle and i>=3:
            update_display(f'> {start:.3f} ~ {end:.3f}({end-start:.3f}), {i}/{n_total}',
                           display_id=handle.display_id)
        else:
            print(f'> {start:.3f} ~ {end:.3f}({end-start:.3f}), {i}/{n_total}')
        if len(ranges) == 1:
            return play_obj

        while play_obj.is_playing():
            time.sleep(0.05)
play_audio.file_path = None
play_audio.audio = None


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


def play_tensor(audio_tensor:torch.Tensor|np.ndarray, play_sec:float=999, sr=16000):
    if isinstance(audio_tensor, np.ndarray):
        audio_tensor = torch.from_numpy(audio_tensor)

    if audio_tensor.dtype == torch.int16:
        audio_tensor = audio_tensor.float() / 32768.0
        # print(audio_tensor.shape, audio_tensor.dtype)

    audio_data = audio_tensor.squeeze().numpy()
    assert audio_data.dtype == np.float32, f"np.float32? {audio_data.dtype}"
    assert audio_data.ndim == 1, f"ndim==1? {audio_data.ndim=} {audio_data.shape=}"
    audio_data = audio_data[:int(16000*play_sec)]
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

def audio_filter(input_audio, af_filter):
    output_audio = Path('/tmp') / Path(input_audio).name
    output_audio = str(output_audio)
    af_filter = af_filter.replace(" ", "").replace("\t", "").replace("\n", "")

    FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
    cmd = f'{FFMPEG} -i {input_audio} -af {af_filter} {output_audio}'
    print(cmd.split())
    subprocess.run(cmd.split(), check=True)
    return output_audio

def audio_to_mp4(audio_path, out_path, af_filter = ""):
    assert Path(audio_path).exists()
    if Path(out_path).exists():
        print(out_path, "already exists")
        return

    if af_filter:
        audio_path = audio_filter(audio_path, af_filter)

    jpg_path = 'testdata/1280x720-bg.jpg'
    FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'

    cmd = f"{FFMPEG} -i {audio_path} -loop 1 -i {jpg_path} -c:v libx264 -c:a aac -b:a 192k -shortest {out_path}"
    print(cmd.split())
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


def compress_mp4(input_path: Path|str, output_path: Path|str,
                 crf: int = 23, preset: str = "medium") -> None:
    """
    Compress an MP4 file using ffmpeg with specified CRF and preset.

    Args:
        input_path (str): Path to the input MP4 file.
        output_path (str): Path where the compressed MP4 file will be saved.
        crf (int): Constant Rate Factor for controlling quality. Lower values mean better quality.
        preset (str): Preset for compression speed. Options include "ultrafast", "superfast", "veryfast",
                      "faster", "fast", "medium", "slow", "slower", "veryslow".

    Returns:
        None

    Raises:
        AssertionError: If the input file does not exist.
    """
    assert Path(input_path).exists(), f"Input file {input_path} does not exist."
    if Path(output_path).exists():
        print(f"{output_path} already exists")
        return

    FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
    cmd = [
        *FFMPEG.split(),
        '-i', input_path,
        *f'-vcodec libx264 -crf {crf}'.split(),
        # '-preset', preset,
        output_path,
    ]
    subprocess.run(cmd, check=True)

# Example usage:
# compress_mp4("input.mp4", "output_compressed.mp4", crf=28, preset="fast")


import subprocess

def get_audio_source(tag = 'a2dp_sink'):
    result = subprocess.run(
        ["pactl", "list", "short", "sources"],
        capture_output=True,
        text=True,
        check=True
    )
    audio_source = [line for line in result.stdout.splitlines() if tag in line]
    audio_source = audio_source[0].split('\t')[1]
    return audio_source


def record_system_audio(output_m4a = "system_out.m4a"):
    output_wav = '/tmp/system_out.wav'
    audio_source = get_audio_source()
    command = [
        *"ffmpeg -nostdin -loglevel warning -threads 0 -y".split(),
        "-f", "pulse",
        "-i", audio_source,
        output_wav
    ]

    try:
        print("Recording... Press Ctrl+C to stop.")
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        command = [
            *"ffmpeg -nostdin -loglevel warning -threads 0".split(),
            "-i", output_wav, "-c:a", "aac", "-b:a", "192k", output_m4a
        ]
        subprocess.run(command, check=True)



def get_chunk(pcm_norm, seg, play_margin:float=0) -> torch.Tensor|np.ndarray:
    assert seg
    s,e = seg.start+play_margin, seg.end+play_margin
    chunk = pcm_norm[:, int(s*16000):int(e*16000)]
    assert chunk.ndim==2 and chunk.shape[1]> 100, chunk.shape
    return chunk


def play_button(pcm_norm:torch.Tensor|np.ndarray, seg:Segment|None,
                *,
                desc:str='play', play_margin:float=0, with_Audio:bool=False):
    from IPython.display import Audio, display
    from ipywidgets import widgets
    if seg:
        desc = f"{desc}: {seg.start:.3f}~{seg.end:.3f}, dur={seg.duration:.3f}"
    if isinstance(pcm_norm, np.ndarray):
        pcm_norm = torch.from_numpy(pcm_norm)

    with_Audio = with_Audio or os.environ.get('SSH_CONNECTION') is not None

    pcm_norm = pcm_norm.type(torch.float32)
    if pcm_norm.ndim == 1:
        pcm_norm = pcm_norm.unsqueeze(0)

    if with_Audio:
        try:
            chunk = get_chunk(pcm_norm, seg, play_margin) if seg else pcm_norm
            display(Audio( chunk, rate=16000))
        except Exception as ex:
            print('\33[91m', ex, '\33[0m')
            print(chunk.shape)
    else:
        button = widgets.Button(description=desc, icon='play',
                                layout=widgets.Layout(width='450px'))
        handle = display(button)
        def play_segment(b, seg:Segment|None):
            try:
                if b.description[-1] == '*': b.description = b.description[:-2]
                chunk = get_chunk(pcm_norm, seg, play_margin) if seg else pcm_norm
                play_tensor( chunk )
                b.description += ' *'
            except Exception as ex:
                print('\33[91m', ex, '\33[0m')
                print(chunk.shape)

        button.on_click(lambda b: play_segment(b, seg))
    return None



def display_audio(wave, label='', sr=16000, ):
    import IPython.display as ipy
    if isinstance(wave, list):
        wave = torch.tensor(wave)
    elif isinstance(wave, np.ndarray):
        wave = torch.from_numpy(wave)
    wave = wave.clone()

    wave[..., 0] = 1
    # wave[..., -1] = 1
    audio_html = ipy.Audio(wave, rate=sr)._repr_html_()
    ipy.display(ipy.HTML(f"<div style='display: flex; align-items: center;'>{audio_html}&nbsp;&nbsp;<pre>{label}</pre></div>"))

#
#
#

def get_chunk(pcm_norm, seg, play_margin:float=0) -> torch.Tensor|np.ndarray:
    assert seg
    assert pcm_norm.ndim==1
    s,e = seg.start+play_margin, seg.end+play_margin
    return pcm_norm[int(s*16000):int(e*16000)]


def play_button(pcm_norm:torch.Tensor|np.ndarray, seg:Segment|None,
                *,
                desc:str='play', play_margin:float=0):
    is_ssh = os.environ.get('SSH_CONNECTION') is not None

    if seg:
        desc = f"{desc}: {seg.start:.3f}~{seg.end:.3f}, dur={seg.duration:.3f}"
    if isinstance(pcm_norm, np.ndarray):
        pcm_norm = torch.from_numpy(pcm_norm)

    pcm_norm = pcm_norm.squeeze().type(torch.float32)

    if is_ssh:
        try:
            chunk = get_chunk(pcm_norm, seg, play_margin) if seg else pcm_norm
            display_audio( chunk, desc)
        except Exception as ex:
            print('\33[91m', ex, '\33[0m')
            print(chunk.shape)
    else:
        import IPython.display as ipy
        from ipywidgets import widgets
        desc = desc.replace('<b>', '').replace('</b>', '')
        desc = desc.replace('<i>', '').replace('</i>', '')
        button = widgets.Button(description=desc, icon='play',
                                layout=widgets.Layout(width='450px'))
        handle = ipy.display(button)
        def play_segment(b, seg:Segment|None):
            try:
                if b.description[-1] == '*': b.description = b.description[:-2]
                chunk = get_chunk(pcm_norm, seg, play_margin) if seg else pcm_norm
                play_tensor( chunk )
                b.description += ' *'
            except Exception as ex:
                print('\33[91m', ex, '\33[0m')
                print(chunk.shape)

        button.on_click(lambda b: play_segment(b, seg))
    return None