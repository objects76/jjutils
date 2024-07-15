import asyncio, os
import vlc # !pip install python-vlc, https://www.olivieraubert.net/vlc/python-ctypes/doc/
import gc, time
import re
from .diar_utils import to_hhmmss
from jjutils.static import static_vars

import simpleaudio
import subprocess, numpy as np
import pydub

from pyannote.core import notebook
import openai

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
    def load_audio(media_input: str, *, sr: int = 16000):
        FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
        try:
            cmd = f"{FFMPEG} -i {media_input} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
            out = subprocess.run(cmd.split(), capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


import pyaudio # pip install PyAudio

FRAME_1ms = 16
class PyAudioPlayer:
    inst = None

    def __init__(self, filename) -> None:
        self.curtime = 0

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format= self.p.get_format_from_width(2),
            channels=1,
            rate= FRAME_1ms*1000,
            start=False,
            frames_per_buffer = FRAME_1ms*5,
            output=True, input=False,
            stream_callback= lambda *args: self.callback(*args))

        self.waves = PyAudioPlayer.load_audio(filename)
        self.beep = pydub.generators.Sine(1200).to_audio_segment(duration=30).apply_gain(-20).raw_data
        self.end_ms = 0
        PyAudioPlayer.inst = self
        pass


    def __del__(self): self.close()

    def close(self):
        self.stream.close()
        self.p.terminate()
        PyAudioPlayer.inst = None

    def callback(self, in_data, frame_count, time_info, status):

        end_frame = self.cur_frame+frame_count
        data = self.wave_chunk[self.cur_frame*2:end_frame*2]
        self.cur_frame += frame_count
        return (data, pyaudio.paContinue)

    def play(self, start_sec, end_sec, use_beep=True):
        # print(f"play: {start_sec}~{end_sec}")
        cur_frame = int(FRAME_1ms * 1000 * start_sec)
        end_frame = int(FRAME_1ms * 1000 * end_sec)

        self.wave_chunk = self.waves[cur_frame*2: end_frame*2]
        if use_beep:
            self.wave_chunk += self.beep

        self.cur_frame = 0

        self.stream.stop_stream()
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    # def get_time(self):
    #     return self.start_sec + (self.cur_frame / FRAME_1ms / 1000.)

    def forward(self, sec:float):
        self.cur_frame += int(FRAME_1ms * 1000 * sec)

    @staticmethod
    def load_audio(media_input: str, *, sr: int = 16000):
        FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
        try:
            cmd = f"{FFMPEG} -i {media_input} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
            out = subprocess.run(cmd.split(), capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten().tobytes()
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

            for ssec, esec, sec_type in ranges:
                if self.stop_play: break
                self.vlcp.set_time(int(ssec * 1000))
                self.end_ms = int(esec * 1000)

                fad_out = sec_type == 's.'
                fad_in = sec_type == 'e.'
                self.audio.play(ssec, -1 if self.play_to_end else esec , use_beep= sec_type != 's.', fade_out=fad_out, fade_in=fad_in)

                while self.stop_play == False and self.audio.play_obj.is_playing():
                    # update text
                    self.vlcp.video_set_marquee_string(
                        vlc.VideoMarqueeOption.Text,
                        self.text + f'\n: {sec_type}{int(self.remained_ms()/1000)} sec remained')
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
    def draw_text(self, text, *, clr_index:int = -1):
        if clr_index < 0:
            clr_index = VlcPlayer.clr_index
            VlcPlayer.clr_index += 1

        clrs = [
            0xFF5733, 0x33FF57, 0x3357FF, 0xFF33A1, 0xFFC300,
            0x8D33FF, 0x33FFF5, 0xFF8D33, 0x57FF33, 0xA1FF33,
        ]
        text_clr = clrs[clr_index % len(clrs)]
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

    def transcribe(self, start_sec, end_sec, *, language='ja', force_update=False, play=False) -> dict:
        clip = self.audio2.data(start_sec, end_sec)
        key = hashlib.sha256(clip).hexdigest()
        if play:
            self.play_obj = simpleaudio.play_buffer(clip, 1, 2, 16_000)

        if key not in HFWhisper.cache or force_update:
            clip = np.frombuffer(clip, np.int16).flatten() / 32768.0

            result = HFWhisper._pipe(clip, return_timestamps=False,
                            generate_kwargs = {
                                "language": f"<|{language}|>",
                                "task":"transcribe"})
            HFWhisper.cache[key] = result

        result = HFWhisper.cache[key]
        for chunk in result.get('chunks', []):
            chunk['timestamp'] = tuple(round(x + start_sec,3) for x in chunk['timestamp'])

        return result


#
#
#
import openai
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

Speaker = namedtuple('Speaker', ['start_sec', 'end_sec', 'speaker_tag'])
class DebugDiarUI:
    _player = None
    def __init__(self, *, video_path = None, offset=0) -> None:
        if DebugDiarUI._player:
            DebugDiarUI._player.clear()

        self.speaker_order = dict()

        self.speaker = None
        self.player = DebugDiarUI._player = VlcPlayer()
        if video_path:
            self.set_videofile(video_path)

        self.prev_end = 0
        self.inter_delay = 0.5

        self.transcribe = True
        self.translate = self.transcribe and True

    def set_videofile(self, video_path):
        assert video_path and Path(video_path).exists()
        self.player.set_file(video_path)
        self.video_path = video_path

        self.whisper = HFWhisper(video_path)
        self.audio_language = 'ja' # 'ko', 'en'


    @singledispatchmethod
    def set_segment(self, anno:Annotation, *, start_sec = 0, min_sec = 0):
        self.rawsegs = []
        for turn, _, speaker_tag in anno.itertracks(yield_label=True):
            if turn.end - turn.start > min_sec:
                self.rawsegs.append( Speaker(turn.start, turn.end, speaker_tag) )
        self.anno = anno
        self.speaker_order = {label: i for i, label in enumerate(self.anno.labels())}

        # find near segment.
        self.cur_segidx = 0
        for i, seg in enumerate(self.rawsegs):
            if seg.start_sec > start_sec: break
            self.cur_segidx = i

        self._setup_ui()

    @set_segment.register
    def _(self, anno_path:str, *, min_sec = 0):
        if anno_path.endswith('.rttm'):
            _, anno = load_rttm(anno_path).popitem()
            self.set_segment(anno, min_sec=min_sec)
        else:
            assert False, f"invalid file format: {anno_path}"

    # @set_segment.register
    # def _(self, rawsegs:list[PklSegment], *, min_sec = 0):
    #     self.rawsegs = [ Speaker(i.start_sec, i.end_sec, i.speaker_tag) for i in rawsegs if i.end_sec - i.start_sec > min_sec]
    #     self._setup_ui()

    def update_caching(self, nth, do_translate=True, force_update=False, debug=False):
        seg = self.rawsegs[nth]
        if seg.speaker_tag.startswith('INTER') or seg.speaker_tag.startswith('OVERLAPPED'):
            return

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
            if not seg.speaker_tag.startswith('SPEAK') and not seg.speaker_tag.startswith('spk'):
                continue
            tag = seg.speaker_tag
            start, end = seg.start_sec, seg.end_sec
            text_ja, text_ko = self.get_text(start,end)
            scripts.append( dict(start=round(start,3), end=round(end,3), speaker=tag, text_ja=text_ja, text_ko=text_ko) )
        return dict(
            media= str(self.video_path),
            stt=type(self.whisper).__name__,
            segments=scripts)

    async def aplay_all(self, segs, slider:widgets.IntSlider):
        print('0. aplay_all done', self.is_playall)
        try:
            seg = segs[slider.value]
            self._play(seg.speaker_tag, seg.start_sec, seg.end_sec)

            while not self.player.assure_play_started(seg.start_sec):
                await asyncio.sleep(0.1)
            # print('wait start', seg.start_sec)

            while not self.player.assure_play_done(seg.start_sec):
                await asyncio.sleep(0.1)
            # print('wait done', seg.start_sec)


            while slider.value < slider.max:
                slider.value = slider.value + 1
                start_sec = segs[slider.value].start_sec

                while not self.player.assure_play_started(start_sec):
                    await asyncio.sleep(0.1)
                # print('wait start', start_sec)

                while not self.player.assure_play_done(start_sec):
                    await asyncio.sleep(0.1)
                # print('wait done', start_sec)

                if not self.is_playall: break
                if self.inter_delay> 0:
                    await asyncio.sleep(self.inter_delay) # interval between segments

        except AttributeError as ex:
            print('aplay_all.ex:', ex)
            pass
        print('1. aplay_all done', self.is_playall)
        self.is_playall = False

    def on_play_all(self, btn, segs, slider:widgets.IntSlider):
        self.is_playall = not self.is_playall
        if self.is_playall:
            btn.icon = 'pause'
            return asyncio.create_task(self.aplay_all(segs, slider))
        else:
            btn.icon = 'play'
            self.player.stop()
            return

    def _play(self, tag, start, end):
        if self.speaker != 'OVERLAPPED' and tag == 'OVERLAPPED':
            return

        self.player.play(start, end)
        self.player.draw_text(
            f"{tag}\ndur={round(end-start,3)} sec, after={round(start-self.prev_end,3)}",
            clr_index= self.speaker_order.get(tag, -1))
        # if tag != 'INTER':
        self.prev_end = end

    def get_text(self, start, end):
        trans = self.whisper.transcribe(start, end, language=self.audio_language)
        text_ja = trans['text']
        text_ko = translate(text_ja) if self.translate else ''
        return (text_ja, text_ko)

    async def a_transcribe(self, seg, details):
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                dur = seg.end_sec - seg.start_sec
                text_ja, text_ko = await loop.run_in_executor(
                    executor,
                    self.get_text, seg.start_sec,seg.end_sec)

                if len(text_ja) > 85 and self.player.play_boundary: text_ja = text_ja[:40] + ' ... ' + text_ja[-40:]
                if len(text_ko) > 85 and self.player.play_boundary: text_ko = text_ko[:40] + ' ... ' + text_ko[-40:]
                # print(text_ja, '\n', text_ko)

                tag = seg.speaker_tag
                range = f"{to_hhmmss(seg.start_sec)} ~ {to_hhmmss(seg.end_sec)} / <span style='color: green;'>{seg.start_sec:.3f} ~ {seg.end_sec:.3f}</span> ({dur:.3f})"
                details.value = f"Range= {range}     <span style='color: red;'>{tag}</span><br>  {type(self.whisper).__name__}: {text_ja}<br>  > {text_ko}"
                return None
        except RuntimeError as ex:
            print('a_transcribe:', ex)

    def _interact_video(self, segs, label='', ui=None):

        count = len(segs)
        # details = widgets.Label(value=f'')
        details = widgets.HTML(value=f'')

        def fn_slider(idx):
            # self.is_playall = False

            seg = segs[idx]
            dur = seg.end_sec - seg.start_sec
            range = f"{to_hhmmss(seg.start_sec)} ~ {to_hhmmss(seg.end_sec)} / <span style='color: green;'>{seg.start_sec:.3f} ~ {seg.end_sec:.3f}</span> ({dur:.3f})"

            text_ja = text_ko = ''
            if self.transcribe and not (
                seg.speaker_tag.startswith('INTER') and seg.speaker_tag.endswith('OVERLAPPED')
                ):
                asyncio.create_task(self.a_transcribe(seg, details))
                text_ja = 'transcribing...'

                # if dur > 10:
                #     text_ja = 'transcribing...'
                #     text_ko = ' '
                #     details.value = f"Range= {range}     <span style='color: red;'>[{seg.speaker_tag}]</span><br>  {text_ja}<br>  {text_ko}"

                # if self.transcribe:
                #     trans = self.whisper.transcribe(seg.start_sec, seg.end_sec)
                #     text_ja = trans['text']
                #     if len(text_ja) > 85: text_ja = text_ja[:40] + ' ... ' + text_ja[-40:]

                # if len(text_ja) and self.translate:
                #     if dur > 10:
                #         text_ko = 'translating...'
                #         details.value = f"Range= {range}     <span style='color: red;'>[{seg.speaker_tag}]</span><br>  {text_ja}<br>  {text_ko}"
                #     text_ko = translate(text_ja)
                #     if len(text_ko) > 85: text_ko = text_ko[:40] + ' ... ' + text_ko[-40:]

            tag = seg.speaker_tag
            details.value = f"Range= {range}     <span style='color: red;'>{tag}</span><br>  {text_ja}<br>  {text_ko}"

            # if self.vlc: self.vlc.stop()
            self._play(seg.speaker_tag, seg.start_sec, seg.end_sec)

        title = widgets.Label(value=f'> {label}: {count= }',
                            style={'background': 'green', 'text_color': 'white'},
                            layout=widgets.Layout(width='400px'),)

        # [0, max]
        slider = widgets.IntSlider(
            value=self.cur_segidx,
            description=f'clips: ',
            min=0, max= max(0, len(segs)-1), step=1,
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            )
        slider.observe(lambda x: fn_slider(x['new']), names='value', type='change')
        def slider_value(val:int):
            if 0<= val < count: slider.value = val

        btn_prev = widgets.Button(description='',  icon='arrow-left', tooltip='' )
        btn_next = widgets.Button(description='', icon='arrow-right', tooltip='' )
        replay_btn = widgets.Button(description='', icon='repeat', tooltip='' )
        playall_btn = widgets.Button(description='', icon='play', tooltip='' )

        replay_btn.on_click(lambda btn: fn_slider(slider.value))
        playall_btn.on_click(lambda btn: self.on_play_all(btn, segs, slider))
        btn_prev.on_click(lambda b: slider_value(slider.value-1))
        btn_next.on_click(lambda b: slider_value(slider.value+1))
        if ui: fn_slider(0) # play initial

        hbox = widgets.HBox([replay_btn, playall_btn, btn_prev, btn_next])
        vbox = widgets.VBox([title, slider, hbox, details, ])
        if ui == None:
            ui = vbox
            display(ui, display_id='11')

        ui.children = vbox.children
        return ui


    def _setup_ui(self):
        SPEAKER_ALL = 'AllSpeaker'
        self.roi_widgets = None
        self.is_playall = False

        self.speaker == SPEAKER_ALL
        self.disp_id = str(time.time())
        display( self.anno, display_id=self.disp_id)
        # update_display( self.anno , display_id=self.disp_id)

        lbl_title = widgets.Label(value= f"[ {self.video_path} ]")
        lbl_title.layout = widgets.Layout(margin='0 0 0 20px')

        def select_speaker(change):
            speaker = change['new']
            self.speaker = speaker

            segs = self.rawsegs if speaker == SPEAKER_ALL else [item for item in self.rawsegs if item.speaker_tag == speaker]
            self.roi_widgets = self._interact_video(segs, f'diar: {speaker}', ui=self.roi_widgets)
            # slider = [w for w in self.roi_widgets.children if isinstance(w, widgets.IntSlider)]

            if speaker == SPEAKER_ALL:
                update_display( self.anno , display_id=self.disp_id)
            else:
                anno = self.anno.label_support(speaker)
                anno.title = speaker
                update_display( anno, display_id=self.disp_id)

        speakers = set( seg.speaker_tag for seg in self.rawsegs)
        speakers = [SPEAKER_ALL, *sorted(speakers)]
        dropdown = widgets.Dropdown(options=speakers, description='Speaker: ')
        dropdown.observe(select_speaker, names='value')
        display(widgets.HBox([dropdown, lbl_title]))
        select_speaker({"new": SPEAKER_ALL})

        # btn_ff5 = widgets.Button(description='+ 5sec')
        # btn_ff10 = widgets.Button(description='+ 10sec')
        # btn_ff5.on_click(lambda b: (self.player.forward(5)))
        # btn_ff10.on_click(lambda b: (self.player.forward(10)))
        # display(widgets.HBox([btn_ff5, btn_ff10]))

        # self._interact_video(self.segs_inter, f'diar: inter')

        self.player.play_boundary = True
        cb_play_boundary = widgets.Checkbox(
            value= self.player.play_boundary,
            description='Play start/end only',
            indent=False
        )
        cb_play_cont = widgets.Checkbox(
            value= False,
            description='Play to end',
            indent=False
        )
        cb_transcribe = widgets.Checkbox(
            value= self.transcribe,
            description='Transcribe',
            indent=False
        )
        cb_translate = widgets.Checkbox(
            value= self.translate,
            description='Translate',
            indent=False,
            disabled = not self.transcribe,
        )

        def on_checkbox_change(change):
            desc, value = change['owner'].description, change['new']
            if desc == 'Play to end':
                self.player.play_to_end = value
            elif desc == 'Play start/end only':
                self.player.play_boundary = value

            elif desc == 'Transcribe':
                self.transcribe = value
                cb_translate.disabled = (value == False)
            elif desc == 'Translate':
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