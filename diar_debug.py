import asyncio, os
import vlc # !pip install python-vlc, https://www.olivieraubert.net/vlc/python-ctypes/doc/
import pyaudio # !pip install PyAudio
import gc, time
import re
from .diar_utils import to_hhmmss
from jjutils.static import static_vars

from typing import Final
import simpleaudio
import subprocess, numpy as np
import pydub
from collections import deque

import logging
logger = logging.getLogger(__name__) # jjutils.diar_debug

from pyannote.core import notebook, Annotation, Segment
from IPython.display import display, Image
def get_openai():
    import openai # pip install openai
    if get_openai._openai is None:
        _openai = openai.OpenAI()
    return get_openai._openai
get_openai._openai = None

os.makedirs('./testdata/cache', exist_ok=True)

class AudioChunk: # for more fine-controlling(ms).
    def __init__(self, mp4path) -> None:
        pcm16k = AudioChunk.load_audio(mp4path)
        self.wave_bytes = pcm16k.tobytes()
        self.duration = len(pcm16k) / 16000
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



# FRAME_1ms = 16
# class PyAudioPlayer:
#     inst = None
#
#     def __init__(self, filename) -> None:
#         self.curtime = 0
#
#         self.p = pyaudio.PyAudio()
#
#         self.stream = self.p.open(
#             format= self.p.get_format_from_width(2),
#             channels=1,
#             rate= FRAME_1ms*1000,
#             start=False,
#             frames_per_buffer = FRAME_1ms*5,
#             output=True, input=False,
#             stream_callback= lambda *args: self.callback(*args))
#
#         self.waves = PyAudioPlayer.load_audio(filename)
#         self.beep = pydub.generators.Sine(1200).to_audio_segment(duration=30).apply_gain(-20).raw_data
#         self.end_ms = 0
#         PyAudioPlayer.inst = self
#         pass
#
#
#     def __del__(self): self.close()
#
#     def close(self):
#         self.stream.close()
#         self.p.terminate()
#         PyAudioPlayer.inst = None
#
#     def callback(self, in_data, frame_count, time_info, status):
#
#         end_frame = self.cur_frame+frame_count
#         data = self.wave_chunk[self.cur_frame*2:end_frame*2]
#         self.cur_frame += frame_count
#         return (data, pyaudio.paContinue)
#
#     def play(self, start_sec, end_sec, use_beep=True):
#         # print(f"play: {start_sec}~{end_sec}")
#         cur_frame = int(FRAME_1ms * 1000 * start_sec)
#         end_frame = int(FRAME_1ms * 1000 * end_sec)
#
#         self.wave_chunk = self.waves[cur_frame*2: end_frame*2]
#         if use_beep:
#             self.wave_chunk += self.beep
#
#         self.cur_frame = 0
#
#         self.stream.stop_stream()
#         self.stream.start_stream()
#
#     def stop(self):
#         self.stream.stop_stream()
#
#     # def get_time(self):
#     #     return self.start_sec + (self.cur_frame / FRAME_1ms / 1000.)
#
#     def forward(self, sec:float):
#         self.cur_frame += int(FRAME_1ms * 1000 * sec)
#
#     @staticmethod
#     def load_audio(media_input: str, *, sr: int = 16000):
#         FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
#         try:
#             cmd = f"{FFMPEG} -i {media_input} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
#             out = subprocess.run(cmd.split(), capture_output=True, check=True).stdout
#             return np.frombuffer(out, np.int16).flatten().tobytes()
#         except subprocess.CalledProcessError as e:
#             raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


#
#
#
async def anullfunc(): pass

HEAD_PLAY = 3 # sec
TAIL_PLAY = 3 # sec

from pathlib import Path
import subprocess
import json

def get_video_resolution(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return stream["width"], stream["height"]

class VlcPlayer:
    def __init__(self, video_path=None,
                 width = 1920*2, height = 1080*2,
                 head_play=None, tail_play=None):
        os.environ["VLC_VERBOSE"] = str("-1")
        os.environ["LIBVA_MESSAGING_LEVEL"] = str("0")

        opts = []
        opts.extend('--video-on-top --no-sub-autodetect-file --no-audio'.split())
        if video_path:
            opts.extend(f'--video-title {Path(video_path).stem}'.split())

        # self.width,self.height = self.vlcp.video_get_size()
        self.width, self.height = get_video_resolution(video_path)
        if self.width <1280*1.5:
            opts.append(f"--zoom={1280.0*1.5/self.width:.1f}")

        self.instance = vlc.Instance(opts)
        self.vlcp: vlc.MediaPlayer = self.instance.media_player_new() # type: ignore
        assert self.vlcp
        self.audio = None
        # self.player.video_set_scale(3)
        self.stop_requested = True
        self.async_play_task = None
        self.text = ''

        self.head_play = head_play or HEAD_PLAY
        self.tail_play = tail_play or TAIL_PLAY
        self.play_boundary = False
        self.play_to_end = False
        self.play_started = []
        self.play_done = []

    def set_file(self, mp4path):
        media:vlc.Media = self.instance.media_new(mp4path)
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
        print('--clear called--')
        # wait not playing...
        self.stop_requested = True
        self.vlcp.stop()
        if self.audio:
            self.audio.stop()

        del self.vlcp
        del self.instance
        del self.audio
        self.audio = self.instance = self.vlcp = None
        gc.collect()

    def stop(self):
        if self.audio:
            self.audio.stop()
        # self.vlcp.pause()

    def stop_auto_play(self):
        self.play_started = []
        self.play_done = []

    in_aplay = False
    requested_segment = (0,0)
    tasks = deque([
        asyncio.create_task(anullfunc()),
        asyncio.create_task(anullfunc())
    ])

    def play(self, start_sec:float, end_sec:float):
        # olds,_ = self.requested_segment
        # if len(self.play_started) > 10: self.play_started.pop(0)
        # self.play_started.append(olds)
        # if len(self.play_started) > 10: self.play_started.pop(0)
        # self.play_started.append(olds)

        self.requested_segment = (start_sec, end_sec)
        if self.tasks[0].done():
            self.tasks.popleft()
            self.tasks.append( asyncio.create_task(self.aplay()) )

    async def aplay(self):
        while self.in_aplay:
            self.stop_requested = True
            await asyncio.sleep(0.1)

        self.in_aplay = True
        start_sec, end_sec = self.requested_segment
        # print(f' aplay({start_sec} ~ {end_sec}), {self.in_aplay}')

        try:
            duration = end_sec - start_sec
            if duration > 8*1.2 and self.play_boundary and not self.play_to_end:
                ranges = [(start_sec, start_sec+self.head_play, 's.'),
                          (end_sec-self.tail_play, end_sec, 'e.')]
            else:
                ranges = [(start_sec, end_sec, '')]
            # beep-end notifier

            # wait playing...
            self.vlcp.play()
            while self.vlcp.get_state() != vlc.State.Playing:
                await asyncio.sleep(0.1)
            # print(' self.vlcp.play() ')
            self.stop_requested = False

            if len(self.play_started) > 10: self.play_started.pop(0)
            logger.debug(f"aplay: + playstarted: {start_sec}")
            self.play_started.append(round(start_sec,3))


            for start_chunk, end_chunk, sec_type in ranges:
                if self.stop_requested: break
                self.vlcp.set_time(int(start_chunk * 1000))
                self.end_ms = int(end_chunk * 1000)

                fad_out = sec_type == 's.'
                fad_in = sec_type == 'e.'
                assert self.audio
                self.audio.play(start_chunk, -1 if self.play_to_end else end_chunk ,
                                use_beep= sec_type != 's.',
                                fade_out=fad_out,
                                fade_in=fad_in)

                while self.stop_requested == False and \
                    self.audio.play_obj and self.audio.play_obj.is_playing():
                    # update text(time remained or current position)
                    current_sec = self.current_ms() / 1000
                    remained_sec = int(self.remained_ms()/1000)
                    self.vlcp.video_set_marquee_string(
                        vlc.VideoMarqueeOption.Text,
                        self.text + f'\n: {sec_type}: cur={current_sec:.1f}, remained={remained_sec}')
                    await asyncio.sleep(0.3) # update text.
                self.audio.stop()
            # del start_chunk, end_chunk, sec_type # for
            if len(self.play_done) > 10: self.play_done.pop(0)
            self.play_done.append(round(start_sec,3))
        except Exception as ex:
            print("ex:", ex)
            pass
        if self.vlcp:
            self.vlcp.pause()
        while self.vlcp.get_state() != vlc.State.Paused: # type: ignore
            await asyncio.sleep(0.1)
        # print(' self.vlcp.pause() ')
        self.in_aplay = False
        pass

    def assure_play_started(self, start_sec:float):
        start_sec = round(start_sec,3)
        return start_sec in self.play_started

    def assure_play_done(self, start_sec:float):
        start_sec = round(start_sec,3)
        return start_sec in self.play_done

    def is_playing(self):
        return self.vlcp and self.vlcp.get_state() == vlc.State.Playing # type: ignore
    # def in_range(self):
    #     return self.remained_ms() > 0
    def remained_ms(self):
        return self.end_ms - self.vlcp.get_time()
    def current_ms(self):
        return self.vlcp.get_time()

    def forward(self, sec:float):
        if self.vlcp.get_state() == vlc.State.Playing: # type: ignore
    # def in_range(self):
            curpos = self.vlcp.get_time()
            self.vlcp.set_time(curpos + sec*1000)
            assert self.audio
            self.audio.forward(sec)

    def _draw_text(self, text, *, clr_argb):
        text_size = int(60*self.width / 1280)
        player = self.vlcp
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Enable, 1)
        player.video_set_marquee_string(vlc.VideoMarqueeOption.Text, text)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Position, 0) # vlc.Position.bottom_right)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Color, clr_argb|0xff000000)  # Red
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Opacity, 255)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Timeout, 0)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Size, text_size)
        self.text = text


    clr_index = 0
    def draw_videotext(self, text, *, clr_index:int = -1, rgba:list[float]|None = None):
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
                                "task":"transcribe"}) # type: ignore

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
        response = get_openai().chat.completions.create(
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
    repr_annotations,
    repr_annotation2
)

from pyannote.core import Annotation, Segment
from collections import namedtuple

SPEAKER_ALL = 'AllSpeaker'
Speaker = namedtuple('Speaker', ['seg', 'track', 'speaker_tag'])

CROP_STEP = 300

from pyannote.core import Annotation, Segment

def get_seginfo(tick: float | Segment, dir:int, anno: Annotation) -> tuple[Segment, str] | None:
    assert anno
    tick_sec = (tick.start + tick.end) / 2 if isinstance(tick, Segment) else tick
    self = get_seginfo
    if anno != self.anno:
        self.anno = anno
        self.tracks = list(anno.itertracks(yield_label=True))

    idx = -1
    for i, (seg,_,lbl) in enumerate(self.tracks):
        if seg.start <= tick_sec <= seg.end:
            idx = i
            break
    if idx < 0:
        if tick_sec != 0:
            print(f'err: no idx for {tick}')
            return None
        idx, dir = 0, 0

    idx += dir
    if idx < 0 or len(self.tracks) <= idx:
        # out of range
        return None

    seg, _, label = self.tracks[idx]
    return seg, label

get_seginfo.tracks = None
get_seginfo.anno = Annotation()


def get_current_seg(tick: float | Segment, anno: Annotation) -> tuple[Segment, str] | None:
    return get_seginfo(tick, 0, anno)

def get_next_seg(tick: float | Segment, anno: Annotation) -> tuple[Segment, str] | None:
    return get_seginfo(tick, 1, anno)

def get_prev_seg(tick: float | Segment, anno: Annotation) -> tuple[Segment, str] | None:
    return get_seginfo(tick, -1, anno)



# called from play_rttm in devutils.py
class DebugDiarUI:
    _player = None
    def __init__(self, *, video_path = None, head_play=None, tail_play=None, transcribe=False) -> None:
        if DebugDiarUI._player:
            DebugDiarUI._player.clear()

        self.speaker_order = dict()

        self.speaker_filter = None
        self.player = DebugDiarUI._player = VlcPlayer(video_path,
                                                      head_play=head_play, tail_play=tail_play)
        if video_path:
            self.set_videofile(video_path)

        self.prev_start = 0
        self.prev_end = 0
        self.inter_delay = 0.5
        self.roi_crop = notebook.crop or Segment(0, self.player.audio.duration)

        self.transcribe = transcribe
        self.translate = self.transcribe and True
        self.rename_history = dict() # stag => ntag

        self.annotations = []
        self.current_seg = Segment(self.roi_crop.start, self.roi_crop.start)


    def set_videofile(self, video_path):
        assert video_path and Path(video_path).exists()
        self.player.set_file(video_path)
        self.video_path = video_path

        self.whisper = None # HFWhisper(video_path)
        self.audio_language = 'ja' # 'ko', 'en'

    def set_annotations(self, annotations:list[Annotation]):
        self.annotations = annotations

    def set_current_anno(self, anno:Annotation, *, start_sec = 0, min_sec = 0):
        self.current_anno = anno
        self.speaker_order = {label: i for i, label in enumerate(self.current_anno.labels())}

        notebook.crop = Segment(
            self.roi_crop.start, min(self.roi_crop.end, self.roi_crop.start + CROP_STEP))

        self._setup_ui()

    def set_segment_from_file(self, anno_path:str, *, min_sec = 0):
        if anno_path.endswith('.rttm'):
            _, anno = load_rttm(anno_path).popitem()
            self.set_current_anno(anno, min_sec=min_sec)
        else:
            assert False, f"invalid file format: {anno_path}"

    @staticmethod
    def is_speaker(tag):
        if tag.startswith('INTER') or tag.startswith('OVERLAPPED'):
            return False
        return True

    def update_caching(self, nth, do_translate=True, force_update=False, debug=False):
        ...
    def caching(self, do_translate=True, force_update=False, debug=False):
        ...
    def get_script(self):
        ...
        return dict()

    async def aplay_all(self):
        logger.debug(f'0. aplay_all enter {self.is_playall}')
        try:
            _seg = Segment(0,0)

            while True:
                self.play_segment(1)
                if _seg == self.current_seg: break
                _seg = self.current_seg

                logger.debug(f'play_all wait started, {_seg}')
                while not self.player.assure_play_started(_seg.start):
                    await asyncio.sleep(0.1)

                logger.debug(f'play_all wait done, {_seg.start}')
                while not self.player.assure_play_done(_seg.start):
                    await asyncio.sleep(0.1)

                if not self.is_playall: break
                if self.inter_delay> 0:
                    await asyncio.sleep(self.inter_delay) # interval between segments


            logger.debug(f'play_all.x: done')

        except AttributeError as ex:
            print('aplay_all.ex:', ex)
            pass
        # print('1. aplay_all done', self.is_playall)
        self.is_playall = False
        logger.debug(f'play_all.x: done, {self.is_playall}')
        self.player.stop_auto_play()




    task_playall = None
    def on_play_all(self, btn):
        self.is_playall = not self.is_playall
        if self.is_playall:
            assert self.task_playall is None
            btn.icon = 'pause'
            btn.description = 'pause'
            self.task_playall = asyncio.create_task(self.aplay_all())
        else:
            btn.icon = 'play'
            btn.description = 'playall'
            self.player.stop()
            if self.task_playall:
                self.task_playall.cancel()
                self.task_playall = None
            return

    def _play(self, segment_speaker, start, end):
        from pyannote.core import notebook
        if self.speaker_filter != 'OVERLAPPED' and segment_speaker == 'OVERLAPPED':
            return
        # self.speaker_filter # speaker filter value in combobox
        assert end-start > 0, f"{start}~{end}, {segment_speaker}"

        self.player.play(start, end)
        self.player.draw_videotext(
            f"{segment_speaker}\ndur={round(end-start,3)} sec, after={round(start-self.prev_end,3)}",
            # clr_index= self.speaker_order.get(tag, -1)
            rgba= notebook[segment_speaker][2] # type: ignore
            )
        # if tag != 'INTER':
        self.prev_start = start
        self.prev_end = end

#     def get_text(self, start, end):
#         if not self.whisper:
#             return ('', '')
#
#         trans = self.whisper.transcribe(start, end, language=self.audio_language)
#         text_ja = trans['text']
#         if self.transcribe and self.audio_language != 'ko':
#             text_ko = translate(text_ja)
#         else:
#             text_ko = ''
#         return (text_ja, text_ko)

#     async def a_transcribe(self, trk, details):
#         if not self.whisper: return
#         import concurrent.futures
#         try:
#             loop = asyncio.get_event_loop()
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 dur = trk.seg.end - trk.seg.start
#
#                 text_ja = text_ko = ''
#                 if self.is_speaker(trk.speaker_tag):
#                     text_ja, text_ko = await loop.run_in_executor(
#                         executor,
#                         self.get_text, trk.start_sec,trk.end_sec)
#                     if len(text_ja) > 85 and self.player.play_boundary: text_ja = text_ja[:40] + ' ... ' + text_ja[-40:]
#                     if len(text_ko) > 85 and self.player.play_boundary: text_ko = text_ko[:40] + ' ... ' + text_ko[-40:]
#                     # print(text_ja, '\n', text_ko)
#
#                 tag = trk.speaker_tag
#                 range = (
#                     f"{to_hhmmss(trk.start_sec)} ~ {to_hhmmss(trk.end_sec)} /"
#                     f" <span style='color: green;'>{trk.start_sec:.3f} ~ {trk.end_sec:.3f}</span> ({dur:.3f})"
#                 )
#                 details.value = (
#                     f"Range= {range}     <span style='color: red;'>{tag}</span>"
#                     # f"<br>  {type(self.whisper).__name__}: {text_ja}<br>  > {text_ko}"
#                 )
#                 return None
#         except RuntimeError as ex:
#             print('a_transcribe:', ex)

    def rename_speakers(self, tags:dict|str):
        if isinstance(tags, dict):
            for from_, to_ in tags.items():
                self.rename_speaker(from_, to_, dump_history=False)
            print(f"#{len(self.rename_history)}: {self.rename_history}")
        elif isinstance(tags, str):
            tagmaps = dict()
            tokens:list[str] = re.split(r'[\s,]+', tags.strip())

            nums = []
            for i in range(len(tokens)):
                if tokens[i].isdecimal():
                    nums.append(int(tokens[i]))
                else:
                    for idx in nums:
                        tagmaps[idx] = f"{tokens[i]},S{idx:02d}"
                    nums = []
            self.rename_speakers(tagmaps)


    def rename_speaker(self, tag:str|int, new_tag:str, dump_history=True):
        if not new_tag: return
        prefix = None
        if isinstance(tag, int):
            for label in self.current_anno.labels():
                if '_' not in label: continue
                prefix = label.split('_')[0]
                tag = f"{prefix}_{tag:02d}"
                break
            if prefix is None: return

        # new_tag = new_tag+tag.replace("SPEAKER_", "_S")
        if tag in self.rename_history and self.rename_history[tag] == new_tag:
            return

        new_trks = []
        changed = False
        for trk in self.roi_tracks:
            if trk.speaker_tag == tag:
                # rename tags in self.roi_tracks
                new_trks.append(Speaker(trk.seg, trk.track, new_tag))
                changed = True
            else:
                new_trks.append(trk)

        if changed:
            assert len(self.roi_tracks) == len(new_trks)
            self.roi_tracks = new_trks
            self.rename_history[tag] = new_tag
            notebook[new_tag] = notebook[tag] # copy style
            if dump_history:
                print(f"#{len(self.rename_history)}: {self.rename_history}")

    def update_crop(self, inc_sec):
        cur = notebook.crop
        notebook.crop = Segment(cur.start+inc_sec,cur.end+inc_sec)
        notebook.crop = notebook.crop & self.roi_crop
        # print(f"new {notebook.crop= }")
        self.update_annos()

        # crop region is changed. so get val in notebook.crop
        if self.current_seg not in notebook.crop:
            for seg, _ in self.current_anno.itertracks():
                if seg in notebook.crop:
                    self.current_seg = seg
                    # print(f"new {self.current_seg= }")
                    break
        pass


    def update_anno_ui(self, trk):
        # restore old segment
        if self.cur_track:
            del self.current_anno[ self.cur_track[0], self.cur_track[1] ]
            self.cur_track = None

        if trk is None: return

        seg,tn,label = trk
        self.cur_track = (seg,'_CUR')
        self.current_anno[ self.cur_track[0], self.cur_track[1] ] = "CUR"

        # readjust notebook.crop
        if seg not in notebook.crop:
            s = max(self.roi_crop.start, seg.start -10)
            e = min(self.roi_crop.end, s+CROP_STEP)
            # print(f"new: {notebook.crop=} -> {s}, {e}")
            notebook.crop = Segment(s,e)
            self.update_annos()
        else:
            self.update_annos(active_only=True)
        # update_display(self.anno, display_id=self.disp_id)
        # for i, ref in enumerate(self.anno_refs):
        #     update_display( ref, display_id=self.disp_id+f"_ref{i}")
        update_display(f"{seg=}, {notebook.crop=}", display_id='log')


    def play_segment(self, direction:int):
        if direction>0:
            trk = get_next_seg(self.current_seg, self.current_anno)
        elif direction<0:
            trk = get_prev_seg(self.current_seg, self.current_anno)
        else:
            trk = get_current_seg(self.current_seg, self.current_anno)
        if trk is None:
            print(f'trk is None, {direction}, {self.current_seg}')
            return

        # print(f"before: {self.current_seg}, {direction=}")

        self.current_seg, speaker_tag = trk
        assert self.current_seg
        assert speaker_tag
        seg = self.current_seg&self.roi_crop
        segstr = f"Segment({seg.start:.3f}, {seg.end:.3f})   {seg.duration:.3f} sec"
        range = f"{to_hhmmss(seg.start)} ~ {to_hhmmss(seg.end)} / <span style='color: green;'>{segstr}</span>"

        self.update_anno_ui((seg,speaker_tag,None))

        text_ja = text_ko = ''
        self.details.value = (f"Range= {range}     "
                              f"<span style='color: red;'>{speaker_tag}</span><br>  "
                              f"{text_ja}<br>  {text_ko}")

        # if self.vlc: self.vlc.stop()
        assert seg.start < seg.end, f"{seg}, {self.current_seg= }, {self.roi_crop}"
        if seg.duration > 0:
            self._play(speaker_tag, seg.start, seg.end)

    def _interact_video(self, label='', ui=None):
        count = len(self.roi_tracks)
        # details = widgets.Label(value=f'')
        details = widgets.HTML(value=f'')
        self.details = details

        title = widgets.Label(value=f'> {label}: {count= }',
                            style={'background': 'green', 'text_color': 'white'},
                            layout=widgets.Layout(width='400px'),)

        # navigation buttons
        btn_prev = widgets.Button(description='prev',  icon='arrow-left', tooltip='' ) # <-
        btn_next = widgets.Button(description='next', icon='arrow-right', tooltip='' ) # ->
        replay_btn = widgets.Button(description='replay', icon='repeat', tooltip='' ) # G
        playall_btn = widgets.Button(description='playall', icon='play', tooltip='' ) # >
        prev_crop = widgets.Button(description='prev roi', icon='angle-double-left', tooltip='')
        next_crop = widgets.Button(description='next roi', icon='angle-double-right', tooltip='')


        replay_btn.on_click(lambda btn: self.play_segment(0))
        playall_btn.on_click(lambda btn: self.on_play_all(btn))
        btn_prev.on_click(lambda b: self.play_segment(-1))
        btn_next.on_click(lambda b: self.play_segment(+1))
        prev_crop.on_click(lambda b: self.update_crop(-CROP_STEP))
        next_crop.on_click(lambda b: self.update_crop(CROP_STEP))


        if ui: self.play_segment(0)

        hbox = widgets.HBox([replay_btn, playall_btn, btn_prev, btn_next, prev_crop, next_crop])
        vbox = widgets.VBox([title, hbox, details, ])
        if ui == None:
            ui = vbox
            display(ui, display_id='11')

        ui.children = vbox.children
        return ui

    def update_annos(self, active_only=False):

        if self.anno_disp_id:
            for i, ref in enumerate(self.annotations):
                if self.current_anno == ref or not active_only:
                    update_display(repr_annotation2(ref, time=True),
                                   display_id=self.anno_disp_id+f"_ref{i}")
        else:
            self.anno_disp_id = str(time.time())
            for i, ref in enumerate(self.annotations):
                if self.current_anno == ref or not active_only:
                    display( repr_annotation2(ref, time=True),
                            display_id=self.anno_disp_id+f"_ref{i}")




    def _select_annotation(self, change):
        new_anno_title = change['new']
        # print('_select_annotation:', new_anno_title)
        if self.current_anno.title == new_anno_title:
            return

        for anno in [self.current_anno, *self.annotations]:
            if anno.title == new_anno_title:
                self.update_anno_ui(None) # clear previous play mark.
                self.current_anno = anno
                self.play_segment(0)
        # self.roi_widgets = self._interact_video(f'diar: {speaker}', ui=self.roi_widgets)
        self.update_annos()


    def _setup_ui(self):
        self.roi_widgets = None
        self.is_playall = False
        self.roi_tracks = []
        self.cur_track = None

        self.speaker_filter == SPEAKER_ALL
        notebook['CUR'] = ("solid", 5, (1,0,1))

        self.anno_disp_id = ""
        self.update_annos()

        # annotation dropdown
        anno_titles = [ref.title for ref in self.annotations]
        dd_annos = widgets.Dropdown(description='Annotation: ',
                                    options=anno_titles, value=anno_titles[-1])
        dd_annos.observe(self._select_annotation, names='value')
        # display(widgets.HBox([dd_annos]), display_id="dropdown_annotation")
        display(dd_annos, display_id="dropdown_annotation")

        # details widget
        self.details = widgets.HTML(
            value="details here",
            placeholder='Details will be shown here',
            description='Details:',
        )
        # display(self.details, display_id="details_widget")
        self._select_annotation({"new": anno_titles[-1]})

        # speaker dropdown
        lbl_title = widgets.Label(value= f"[ {self.video_path} ]")
        lbl_title.layout = widgets.Layout(margin='0 0 0 20px')

        def select_speaker(change):
            speaker = change['new']
            self.speaker_filter = speaker
            # if speaker == SPEAKER_ALL:
            #     self.roi_tracks = self.raw_tracks
            # else:
            #     self.roi_tracks = [item for item in self.raw_tracks if item.speaker_tag == speaker]
            self.roi_widgets = self._interact_video(f'diar: {speaker}', ui=self.roi_widgets)
            # slider = [w for w in self.roi_widgets.children if isinstance(w, widgets.IntSlider)]
            self.update_annos()
            # update_display( self.anno , display_id=self.disp_id)
            # for i, ref in enumerate(self.anno_refs):
            #     update_display(ref, display_id=self.disp_id+f"_ref{i}")

        speakers = self.current_anno.labels() # set( seg.speaker_tag for seg in self.raw_tracks)
        speakers = [SPEAKER_ALL, *sorted(speakers)]
        dd_speaker = widgets.Dropdown(options=speakers, description='Speaker: ')
        dd_speaker.observe(select_speaker, names='value')
        display(widgets.HBox([dd_speaker, lbl_title]), display_id="dropdown_speaker")
        select_speaker({"new": SPEAKER_ALL})

        # btn_ff5 = widgets.Button(description='+ 5sec')
        # btn_ff10 = widgets.Button(description='+ 10sec')
        # btn_ff5.on_click(lambda b: (self.player.forward(5)))
        # btn_ff10.on_click(lambda b: (self.player.forward(10)))
        # display(widgets.HBox([btn_ff5, btn_ff10]))

        # self._interact_video(self.segs_inter, f'diar: inter')
        BTN_PLAY_BOUNDARY:Final[str] = 'Play start&end only'
        BTN_PLAY_TO_END:Final[str] = 'Play to end'
        BTN_TRANSCRIBE:Final[str] = 'Transcribe'
        BTN_TRANSLATE:Final[str] = 'Translate'

        self.player.play_boundary = True
        cb_play_boundary = widgets.Checkbox(
            value= self.player.play_boundary,
            description=BTN_PLAY_BOUNDARY,
            indent=False
        )
        cb_play_cont = widgets.Checkbox(
            value= False,
            description=BTN_PLAY_TO_END,
            indent=False
        )
        cb_transcribe = widgets.Checkbox(
            value= self.transcribe,
            description=BTN_TRANSCRIBE,
            indent=False
        )
        cb_translate = widgets.Checkbox(
            value= self.translate,
            description=BTN_TRANSLATE,
            indent=False,
            disabled = not self.transcribe,
        )

        def on_checkbox_change(change):
            desc, value = change['owner'].description, change['new']
            if desc == BTN_PLAY_TO_END:
                self.player.play_to_end = value
            elif desc == BTN_PLAY_BOUNDARY:
                self.player.play_boundary = value

            elif desc == BTN_TRANSCRIBE:
                self.transcribe = value
                cb_translate.disabled = (value == False)
            elif desc == BTN_TRANSLATE:
                self.translate = value

        cb_play_boundary.observe(on_checkbox_change, names='value')
        cb_play_cont.observe(on_checkbox_change, names='value')
        cb_transcribe.observe(on_checkbox_change, names='value')
        cb_translate.observe(on_checkbox_change, names='value')

        def select_inter_delay(change):
            delay = change['new']
            self.inter_delay = float(delay)
        inter_delay = widgets.Dropdown(options='0.0,0.5,1.0,1.5'.split(','),
                                       value=str(self.inter_delay), description='delay(sec): ', )
        inter_delay.observe(select_inter_delay, names='value')

        btn_close = widgets.Button(description='Close')
        btn_close.on_click(lambda b: (self.player.clear()))
        hbox = widgets.HBox([cb_play_boundary, cb_play_cont, cb_transcribe, cb_translate, inter_delay])
        display(hbox, btn_close)
        pass

# dui = DebugUI(mp4file)
# dui.set_segment(no_shortvoice_segs)


#
#
#
from pydub import AudioSegment
from pydub import playback
import io

@static_vars(cache = shelve.open('./testdata/cache/tts.shelve'))
def tts(text:str, man=False):
    if text not in tts.cache:
        response = get_openai().audio.speech.create(
            model="tts-1", # tts-1-hd
            voice="onyx" if man else 'nova', # 'nova', onyx, https://platform.openai.com/docs/guides/text-to-speech
            response_format = 'mp3', speed=1.,
            input=text,
        )
        tts.cache[text] = response.content

    mp3bytes = tts.cache[text]
    audio_segment = AudioSegment.from_file(io.BytesIO(mp3bytes), format="mp3")
    playback.play( audio_segment )