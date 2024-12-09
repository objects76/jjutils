import asyncio, os
import vlc # !pip install python-vlc, https://www.olivieraubert.net/vlc/python-ctypes/doc/
import pyaudio # !pip install PyAudio
import gc, time
import re
from .diar_utils import to_hhmmss
from jjutils.static import static_vars

import simpleaudio
import subprocess, numpy as np
import pydub

from pyannote.core import notebook
import openai # pip install openai

_openai = openai.OpenAI()
os.makedirs('./testdata/cache', exist_ok=True)

class AudioChunk: # for more fine-controlling(ms).
    def __init__(self, mp4path) -> None:
        self.wave_bytes = AudioChunk.load_audio(mp4path).tobytes()
        self.play_obj = None

        BEEP_FREQ = 1200
        self.beep = pydub.generators.Sine(BEEP_FREQ).to_audio_segment(duration=30).apply_gain(-30).raw_data
        pass

    def play(self, start_sec, end_sec, use_beep = True, fade_in=False, fade_out=False, wait_done=False):
        self.stop()

        wave_clip = self.data(start_sec, end_sec)
        if use_beep:
            wave_clip += self.beep

        if fade_in or fade_out:
            audio_seg = pydub.AudioSegment(wave_clip, sample_width=2, frame_rate=16_000, channels=1)
            if fade_in:
                audio_seg = audio_seg.fade(from_gain=-20, duration=1200, start=0)# fade_in
            if fade_out:
                audio_seg = audio_seg.fade(to_gain=-20, duration=1200, end=float('inf')) # fade_out
            wave_clip = audio_seg.raw_data
        self.play_obj = simpleaudio.play_buffer(wave_clip, 1, 2, 16_000)
        if wait_done:
            self.play_obj.wait_done()
            self.play_obj = None


    def data(self, start_sec, end_sec):
        assert (start_sec < end_sec) or (end_sec <= 0)
        sample_start = int(start_sec * 16_000)
        sample_end = int(end_sec * 16_000)
        if sample_end <= 0:
            return self.wave_bytes[sample_start*2:]
        else:
            return self.wave_bytes[sample_start*2:sample_end*2]

    def wait(self):
        if self.play_obj:
            self.play_obj.wait_done()

    def stop(self):
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None

    def forward(self, sec:float):
        ...

    # @staticmethod
    # def apply_fade_out(audio_segment, fade_out_duration_ms=1000):
    #     # Apply fade-out effect
    #     faded_audio = audio_segment.fade_out(fade_out_duration_ms)
    #     return faded_audio

    # @staticmethod
    # def apply_fade_in(audio_segment, fade_in_duration_ms=1000):
    #     faded_audio = audio_segment.fade_in(fade_in_duration_ms)
    #     return faded_audio


    def plot(self, start_sec, end_sec):
        signal = self.data(start_sec, end_sec)
        signal = np.fromstring(signal, np.int16)
        Time = np.linspace(0, len(signal) / 16_000, num=len(signal)) + start_sec

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(notebook.width, 3))
        ax = fig.gca()
        unit = round((end_sec-start_sec) / 30,1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(unit)) # jjkim
        ax.axes.get_yaxis().set_visible(False)
        plt.title("Signal Wave...")
        plt.plot(Time, signal)
        plt.show()


    @staticmethod
    def load_audio(media_input: str, *, sr: int = 16000) -> np.ndarray:
        FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
        try:
            cmd = f"{FFMPEG} -i {media_input} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
            out = subprocess.run(cmd.split(), capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e



#
#
#
class VlcPlayer:
    def __init__(self, width = 1920*2, height = 1080*2):
        os.environ["VLC_VERBOSE"] = str("-1")

        opts = []
        opts.extend('--video-on-top --no-sub-autodetect-file --no-audio'.split())
        # opts.extend(f"--video-on-top --width={width} --height={height}".split())

        self.instance = vlc.Instance(opts)
        self.vlcp: vlc.MediaPlayer = self.instance.media_player_new()

        # self.player.video_set_scale(3)
        self.stop_play = True
        self.async_play_task = None
        self.text = ''
        self.n_played = 0
        self.play_boundary = False
        self.play_to_end = False
        self.play_started = []
        self.play_done = []


    def set_file(self, mp4path):
        media = self.instance.media_new(mp4path)
        self.vlcp.set_media(media)

        # self.vlcp.audio_set_mute(True)
        self.audio = AudioChunk(mp4path)
        # self.audio = PyAudioPlayer(mp4path)

        media.parse()
        width,height = self.vlcp.video_get_size()
        # scale = 1.5
        # if width*height <= 640*360: scale = 3
        # self.player.video_set_scale(scale)


        # for track in media.tracks_get():
        #     track:vlc.MediaTrack
        #     if track.type == vlc.TrackType.video:
        #         print(type(track.video))
        #         # print(f"Video resolution: {track.video.width} x {track.video.height}")


    def __del__(self): self.clear()

    def clear(self):
        if self.instance == None: return
        # wait not playing...
        self.stop_play = True
        self.vlcp.stop()
        self.audio.stop()

        del self.vlcp
        del self.instance
        del self.audio
        self.audio = self.instance = self.vlcp = None
        gc.collect()

    def stop(self):
        self.audio.stop()
        # self.vlcp.pause()

    def play(self, start_sec:float, end_sec:float):
        asyncio.create_task(self.aplay(start_sec, end_sec))

    async def aplay(self, start_sec:float, end_sec:float):

        self.n_played += 1
        try:
            # wait not playing...
            while self.n_played >= 2:
                self.stop_play = True
                await asyncio.sleep(0.1)

            duration = end_sec - start_sec
            if duration > 8*1.2 and self.play_boundary and not self.play_to_end:
                ranges = [(start_sec, start_sec+4, 's.'), (end_sec-3, end_sec, 'e.')]
            else:
                ranges = [(start_sec, end_sec, '')]
            # beep-end notifier

            # wait playing...
            self.vlcp.play()
            while self.vlcp.get_state() != vlc.State.Playing:
                await asyncio.sleep(0.1)
            self.stop_play = False

            if len(self.play_started) > 5: self.play_started.pop(0)
            self.play_started.append(start_sec)

            for start_sec, end_sec, sec_type in ranges:
                if self.stop_play: break
                self.vlcp.set_time(int(start_sec * 1000))
                self.end_ms = int(end_sec * 1000)

                fad_out = sec_type == 's.'
                fad_in = sec_type == 'e.'
                self.audio.play(start_sec, -1 if self.play_to_end else end_sec , use_beep= sec_type != 's.', fade_out=fad_out, fade_in=fad_in)

                while self.stop_play == False and self.audio.play_obj.is_playing():
                    # update text(time remained or current position)
                    current_sec = self.current_ms() / 1000
                    remained_sec = int(self.remained_ms()/1000)
                    self.vlcp.video_set_marquee_string(
                        vlc.VideoMarqueeOption.Text,
                        self.text + f'\n: {sec_type}: cur={current_sec:.1f}, remained={remained_sec}')
                    await asyncio.sleep(0.3) # update text.
                self.audio.stop()
            # for
            if len(self.play_done) > 5: self.play_done.pop(0)
            self.play_done.append(start_sec)


        except AttributeError:
            pass

        if self.vlcp and self.n_played <= 1:
            self.vlcp.pause()
        self.n_played -= 1

    def assure_play_started(self, start_sec:float):
        return start_sec in self.play_started

    def assure_play_done(self, start_sec:float):
        return start_sec in self.play_done

    def is_playing(self):
        return self.vlcp and self.vlcp.get_state() == vlc.State.Playing
    # def in_range(self):
    #     return self.remained_ms() > 0
    def remained_ms(self):
        return self.end_ms - self.vlcp.get_time()
    def current_ms(self):
        return self.vlcp.get_time()

    def forward(self, sec:float):
        if self.vlcp.get_state() == vlc.State.Playing:
            curpos = self.vlcp.get_time()
            self.vlcp.set_time(curpos + sec*1000)
            self.audio.forward(sec)

    def _draw_text(self, text, *, clr_argb):
        player = self.vlcp
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Enable, 1)
        player.video_set_marquee_string(vlc.VideoMarqueeOption.Text, text)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Position, 0) # vlc.Position.bottom_right)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Color, clr_argb|0xff000000)  # Red
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Opacity, 255)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Timeout, 0)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Size, 60)
        self.text = text


    clr_index = 0
    def draw_text(self, text, *, clr_index:int = -1, rgba:list[float] = None):
        if rgba is None:
            if clr_index < 0:
                clr_index = VlcPlayer.clr_index
                VlcPlayer.clr_index += 1

            clrs = [
                0xFF5733, 0x33FF57, 0x3357FF, 0xFF33A1, 0xFFC300,
                0x8D33FF, 0x33FFF5, 0xFF8D33, 0x57FF33, 0xA1FF33,
            ]
            text_clr = clrs[clr_index % len(clrs)]
        else:
            r_hex = (int(rgba[0] * 255)&0xff) << 16
            g_hex = (int(rgba[1] * 255)&0xff) << 8
            b_hex = (int(rgba[2] * 255)&0xff) << 0
            text_clr = r_hex | g_hex | b_hex

        self._draw_text(text, clr_argb=text_clr)


#
#
#
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import simpleaudio
from jjutils.diar_debug import AudioChunk
import hashlib
import shelve




class HFWhisper:

    _mp4file = None
    _whisper = None
    @staticmethod
    def transcribe_(mp4file, start_sec, end_sec, *, language,force_update=False):
        self = HFWhisper
        if self._mp4file != mp4file:
            self._whisper = HFWhisper(mp4file)
            self._mp4file = mp4file

        return self._whisper.transcribe(start_sec, end_sec, language=language,force_update=force_update)


    _pipe = None
    cache = shelve.open('./testdata/cache/whisper.shelve')

    def __init__(self, mp4file) -> None:
        self.audio2 = AudioChunk(mp4file)

        dtype = torch.bfloat16
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, # low_cpu_mem_usage=True, use_safetensors=True
        )

        if HFWhisper._pipe is None:
            processor = AutoProcessor.from_pretrained(model_id)
            HFWhisper._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                # max_new_tokens=128,
                # chunk_length_s=30,
                # batch_size=16,
                # return_timestamps=True,
                torch_dtype=dtype,
                device='cuda:0',
            )

    def __del__(self):
        if HFWhisper._pipe:
            del HFWhisper._pipe
            torch.cuda.empty_cache()
            HFWhisper._pipe = None

    def play(self, start_sec, end_sec):
        self.audio2.play(start_sec, end_sec)

    def transcribe(self, start_sec, end_sec, *, language, force_update=False, play=False) -> dict:
        clip = self.audio2.data(start_sec, end_sec)
        key = hashlib.sha256(clip).hexdigest()
        if play:
            self.play_obj = simpleaudio.play_buffer(clip, 1, 2, 16_000)

        if key not in HFWhisper.cache or force_update:
            clip_norm = np.frombuffer(clip, np.int16).flatten() / 32768.0

            result = HFWhisper._pipe(clip_norm, return_timestamps=False,
                            generate_kwargs = {
                                "language": f"<|{language}|>",
                                "task":"transcribe"})

            HFWhisper.cache[key] = result

        result = HFWhisper.cache[key]
        for chunk in result.get('chunks', []):
            chunk['timestamp'] = tuple(round(x + start_sec,3) for x in chunk['timestamp'])

        class WaveObj:
            def __init__(self, clip): self.clip = clip
            def play(self): return simpleaudio.play_buffer(self.clip, 1, 2, 16_000)

        result['audio'] = WaveObj(clip)
        return result


#
#
#
import shelve
import json

@static_vars(cache = shelve.open('./testdata/cache/translate.shelve'))
def translate(text_ja, force_update = False):
    import openai
    if text_ja not in translate.cache or force_update:
        # print('openai called')
        user_msg = dict(role='user', content=(
            "Translate the following Japanese to Korean:"
            + "\n"
            + f"\nJapanese: {text_ja}"
            + "\n"
            + "\nOutput format example:"
            + "\n{ \"kor\": \"translated korean is here...\"}"
        ))
        response = _openai.chat.completions.create(
            model="gpt-4o",  # or the latest model you want to use
            messages=[user_msg],
            temperature = 0,
            top_p = 0,
        )
        try:
            result = response.choices[0].message.content
            result = result.replace('```json', '').replace('```', '')
            text_ko = json.loads(result)['kor']
        except:
            text_ko = response.choices[0].message.content

        translate.cache[text_ja] = text_ko
    try:
        return translate.cache[text_ja]
    except:
        return text_ja

#
#
#
from typing import cast, Iterator
from pathlib import Path
from pyannote.database.util import load_rttm
from IPython.display import display, update_display
import ipywidgets as widgets
from functools import singledispatch, singledispatchmethod
from jjutils.diar_utils import (
    annotation_to_pklsegments
)

from pyannote.core import Annotation, Segment
from collections import namedtuple
import time
import threading
import asyncio
from pathlib import Path
from collections import namedtuple
from typing import cast, Iterator

import tkinter as tk
from tkinter import ttk

# Assuming these are defined elsewhere in your code
# from pyannote.core import Annotation, Segment
# from your_player_module import VlcPlayer
# from your_transcription_module import HFWhisper, translate
# from your_utils_module import to_hhmmss

Speaker = namedtuple('Speaker', ['start_sec', 'end_sec', 'speaker_tag'])

class DebugDiarUI:
    _player = None

    def __init__(self, *, video_path=None, transcribe=False, offset=0) -> None:
        if DebugDiarUI._player:
            DebugDiarUI._player.clear()

        self.speaker_order = dict()
        self.speaker_filter = None
        self.player = DebugDiarUI._player = VlcPlayer()
        if video_path:
            self.set_videofile(video_path)

        self.prev_end = 0
        self.inter_delay = 0.5

        self.transcribe = transcribe
        self.translate = self.transcribe and True
        self.roi_widgets = None

    def set_videofile(self, video_path):
        assert video_path and Path(video_path).exists()
        self.player.set_file(video_path)
        self.video_path = video_path

        self.whisper = HFWhisper(video_path)
        self.audio_language = 'ja'  # 'ko', 'en'

    def set_segment(self, anno, *, start_sec=0, min_sec=0):
        self.rawsegs = []
        iter_tracks = cast(
            Iterator[tuple[Segment, str, str]],
            anno.itertracks(yield_label=True))

        for turn, _, speaker_tag in iter_tracks:
            if turn.end - turn.start > min_sec:
                self.rawsegs.append(Speaker(turn.start, turn.end, speaker_tag))
        self.anno = anno
        self.speaker_order = {label: i for i, label in enumerate(self.anno.labels())}

        # Find nearest segment.
        self.cur_segidx = 0
        for i, seg in enumerate(self.rawsegs):
            if seg.start_sec > start_sec:
                break
            self.cur_segidx = i

        self._setup_ui()

    @staticmethod
    def is_speaker(tag):
        if tag.startswith('INTER') or tag.startswith('OVERLAPPED'):
            return False
        return True

    def update_caching(self, nth, do_translate=True, force_update=False, debug=False):
        seg = self.rawsegs[nth]
        text_ja = text_ko = ''
        if self.is_speaker(seg.speaker_tag):
            trans = self.whisper.transcribe(seg.start_sec, seg.end_sec, force_update=force_update, language=self.audio_language)
            text_ja = trans['text']
            text_ko = ''
            if self.audio_language != 'ko' and do_translate:
                text_ko = translate(text_ja, force_update)
            if debug:
                print(f'{seg.start_sec}~{seg.end_sec}:\n{text_ja}\n{text_ko}')

    def caching(self, do_translate=True, force_update=False, debug=False):
        for ith in range(len(self.rawsegs)):
            self.update_caching(ith, do_translate=do_translate, force_update=force_update, debug=debug)

    def get_script(self):
        scripts = []
        for seg in self.rawsegs:
            text_ja = text_ko = ''
            if self.is_speaker(seg.speaker_tag):
                tag = seg.speaker_tag
                start, end = seg.start_sec, seg.end_sec
                text_ja, text_ko = self.get_text(start, end)
            scripts.append(dict(start=round(start, 3), end=round(end, 3), speaker=tag, text_ja=text_ja, text_ko=text_ko))
        return dict(
            media=str(self.video_path),
            stt=type(self.whisper).__name__,
            segments=scripts)

    def _play(self, segment_speaker, start, end):
        if self.speaker_filter != 'OVERLAPPED' and segment_speaker == 'OVERLAPPED':
            return

        self.player.play(start, end)
        self.player.draw_text(
            f"{segment_speaker}\ndur={round(end - start, 3)} sec, after={round(start - self.prev_end, 3)}",
            rgba=(255, 0, 0, 128)  # Placeholder for color
        )
        self.prev_end = end

    def get_text(self, start, end):
        trans = self.whisper.transcribe(start, end, language=self.audio_language)
        text_ja = trans['text']
        if self.transcribe and self.audio_language != 'ko':
            text_ko = translate(text_ja)
        else:
            text_ko = ''
        return (text_ja, text_ko)

    def a_transcribe(self, seg):
        dur = seg.end_sec - seg.start_sec
        text_ja = text_ko = ''
        if self.is_speaker(seg.speaker_tag):
            text_ja, text_ko = self.get_text(seg.start_sec, seg.end_sec)
            if len(text_ja) > 85 and self.player.play_boundary:
                text_ja = text_ja[:40] + ' ... ' + text_ja[-40:]
            if len(text_ko) > 85 and self.player.play_boundary:
                text_ko = text_ko[:40] + ' ... ' + text_ko[-40:]

        tag = seg.speaker_tag
        range_text = (
            f"{to_hhmmss(seg.start_sec)} ~ {to_hhmmss(seg.end_sec)} /"
            f" {seg.start_sec:.3f} ~ {seg.end_sec:.3f} ({dur:.3f})"
        )
        details_text = (
            f"Range= {range_text}     [{tag}]\n"
            f"  {type(self.whisper).__name__}: {text_ja}\n  > {text_ko}"
        )
        # Update the UI in the main thread
        self.details_label.config(text=details_text)

    def _interact_video(self, segs, label='', ui=None):
        count = len(segs)

        # Create a new window or frame for the UI
        if ui is None:
            ui = tk.Toplevel(self.root)
            ui.title(label)
            self.ui = ui
        else:
            # Clear existing UI elements
            for widget in ui.winfo_children():
                widget.destroy()

        # Title
        title_label = tk.Label(ui, text=f'> {label}: count= {count}')
        title_label.pack()

        # Slider
        self.slider_var = tk.IntVar(value=self.cur_segidx)
        slider = tk.Scale(ui, from_=0, to=max(0, len(segs) - 1), orient='horizontal', variable=self.slider_var, command=self.on_slider_change)
        slider.pack()

        # Buttons
        btn_prev = tk.Button(ui, text='Prev', command=lambda: self.slider_value(self.slider_var.get() - 1))
        btn_next = tk.Button(ui, text='Next', command=lambda: self.slider_value(self.slider_var.get() + 1))
        replay_btn = tk.Button(ui, text='Replay', command=lambda: self.fn_slider(self.slider_var.get()))
        playall_btn = tk.Button(ui, text='Play All', command=lambda: self.on_play_all(playall_btn, segs, self.slider_var))

        btn_frame = tk.Frame(ui)
        btn_frame.pack()
        btn_prev.pack(in_=btn_frame, side='left')
        btn_next.pack(in_=btn_frame, side='left')
        replay_btn.pack(in_=btn_frame, side='left')
        playall_btn.pack(in_=btn_frame, side='left')

        # Details label
        self.details_label = tk.Label(ui, text='', justify='left', anchor='w')
        self.details_label.pack()

        # Set up variables
        self.segs = segs
        self.slider = slider

        return ui

    def fn_slider(self, idx):
        seg = self.segs[idx]
        dur = seg.end_sec - seg.start_sec
        start_hhmmss = to_hhmmss(seg.start_sec)
        end_hhmmss = to_hhmmss(seg.end_sec)
        range_text = f"{start_hhmmss} ~ {end_hhmmss} / {seg.start_sec:.3f} ~ {seg.end_sec:.3f} ({dur:.3f})"

        text_ja = text_ko = ''
        if self.transcribe and not (seg.speaker_tag.startswith('INTER') and seg.speaker_tag.endswith('OVERLAPPED')):
            # Start transcription in a separate thread
            threading.Thread(target=self.a_transcribe, args=(seg,)).start()
            text_ja = 'transcribing...'

        tag = seg.speaker_tag
        self.details_label.config(text=f"Range= {range_text}     [{tag}]\n  {text_ja}\n  {text_ko}")

        # Play the segment
        self._play(seg.speaker_tag, seg.start_sec, seg.end_sec)

    def on_slider_change(self, value):
        idx = int(value)
        self.fn_slider(idx)

    def slider_value(self, val: int):
        if 0 <= val < len(self.segs):
            self.slider_var.set(val)

    def on_play_all(self, btn, segs, slider_var):
        self.is_playall = not self.is_playall
        if self.is_playall:
            btn.config(text='Pause')
            threading.Thread(target=self.play_all_segments, args=(segs, slider_var)).start()
        else:
            btn.config(text='Play All')
            self.player.stop()

    def play_all_segments(self, segs, slider_var):
        try:
            idx = slider_var.get()
            while idx < len(segs):
                seg = segs[idx]
                self._play(seg.speaker_tag, seg.start_sec, seg.end_sec)
                while not self.player.assure_play_started(seg.start_sec):
                    time.sleep(0.1)
                while not self.player.assure_play_done(seg.start_sec):
                    time.sleep(0.1)
                if not self.is_playall:
                    break
                if self.inter_delay > 0:
                    time.sleep(self.inter_delay)
                idx += 1
                slider_var.set(idx)
        except Exception as ex:
            print('play_all_segments exception:', ex)
        finally:
            self.is_playall = False
            btn.config(text='Play All')

    def on_checkbox_change(self):
        self.player.play_to_end = self.play_to_end_var.get()
        self.player.play_boundary = self.play_boundary_var.get()
        self.transcribe = self.transcribe_var.get()
        self.translate = self.translate_var.get()
        if not self.transcribe:
            # Disable translate checkbox
            self.translate_var.set(False)
            self.translate_checkbutton.config(state='disabled')
        else:
            self.translate_checkbutton.config(state='normal')

    def select_speaker(self, event=None):
        speaker = self.speaker_var.get()
        self.speaker_filter = speaker

        segs = self.rawsegs if speaker == 'AllSpeaker' else [item for item in self.rawsegs if item.speaker_tag == speaker]

        # Check if roi_widgets exists
        if hasattr(self, 'roi_widgets'):
            self.roi_widgets = self._interact_video(segs, f'diar: {speaker}', ui=self.roi_widgets)
        else:
            self.roi_widgets = self._interact_video(segs, f'diar: {speaker}')

    def select_inter_delay(self, event=None):
        delay = self.inter_delay_var.get()
        self.inter_delay = float(delay)

    def on_close(self):
        self.player.clear()
        self.root.destroy()

    def _setup_ui(self):
        SPEAKER_ALL = 'AllSpeaker'
        self.speaker_filter = SPEAKER_ALL
        self.roi_widgets = None

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Debug Diarization UI")

        lbl_title = tk.Label(self.root, text=f"[ {self.video_path} ]")
        lbl_title.pack()

        # Dropdown for selecting speaker
        speakers = set(seg.speaker_tag for seg in self.rawsegs)
        speakers = [SPEAKER_ALL, *sorted(speakers)]
        self.speaker_var = tk.StringVar(value=SPEAKER_ALL)
        dropdown_label = tk.Label(self.root, text='Speaker:')
        dropdown_label.pack()
        dropdown = ttk.Combobox(self.root, textvariable=self.speaker_var, values=speakers)
        dropdown.pack()
        dropdown.bind("<<ComboboxSelected>>", self.select_speaker)

        # Initialize speaker selection
        self.select_speaker()

        # Checkbuttons for various options
        self.play_boundary_var = tk.BooleanVar(value=self.player.play_boundary)
        cb_play_boundary = tk.Checkbutton(self.root, text='Play start/end only', variable=self.play_boundary_var, command=self.on_checkbox_change)
        cb_play_boundary.pack()

        self.play_to_end_var = tk.BooleanVar(value=False)
        cb_play_cont = tk.Checkbutton(self.root, text='Play to end', variable=self.play_to_end_var, command=self.on_checkbox_change)
        cb_play_cont.pack()

        self.transcribe_var = tk.BooleanVar(value=self.transcribe)
        cb_transcribe = tk.Checkbutton(self.root, text='Transcribe', variable=self.transcribe_var, command=self.on_checkbox_change)
        cb_transcribe.pack()

        self.translate_var = tk.BooleanVar(value=self.translate)
        self.translate_checkbutton = tk.Checkbutton(self.root, text='Translate', variable=self.translate_var, command=self.on_checkbox_change)
        self.translate_checkbutton.pack()
        if not self.transcribe:
            self.translate_checkbutton.config(state='disabled')

        # Dropdown for inter delay
        inter_delay_options = ['0.0', '0.5', '1.0', '1.5']
        self.inter_delay_var = tk.StringVar(value=str(self.inter_delay))
        inter_delay_label = tk.Label(self.root, text='Delay (sec):')
        inter_delay_label.pack()
        inter_delay_dropdown = ttk.Combobox(self.root, textvariable=self.inter_delay_var, values=inter_delay_options)
        inter_delay_dropdown.pack()
        inter_delay_dropdown.bind("<<ComboboxSelected>>", self.select_inter_delay)

        # Close button
        btn_close = tk.Button(self.root, text='Close', command=self.on_close)
        btn_close.pack()

        # Start the Tkinter main loop
        self.root.mainloop()


#
#
#
from pydub import AudioSegment
from pydub import playback
import io



@static_vars(cache = shelve.open('./testdata/cache/tts.shelve'))
def tts(text:str, man=False):
    if text not in tts.cache:
        response = _openai.audio.speech.create(
            model="tts-1", # tts-1-hd
            voice="onyx" if man else 'nova', # 'nova', onyx, https://platform.openai.com/docs/guides/text-to-speech
            response_format = 'mp3', speed=1.,
            input=text,
        )
        tts.cache[text] = response.content

    mp3bytes = tts.cache[text]
    audio_segment = AudioSegment.from_file(io.BytesIO(mp3bytes), format="mp3")
    playback.play( audio_segment )