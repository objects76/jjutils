import asyncio, os
import vlc # !pip install python-vlc, https://www.olivieraubert.net/vlc/python-ctypes/doc/
import gc, time
import re
from .diar_utils import to_hhmmss


import simpleaudio
import subprocess, numpy as np
import pydub

class AudioChunk: # for more fine-controlling(ms).
    def __init__(self, mp4path) -> None:
        self.wave_bytes = AudioChunk.load_audio(mp4path).tobytes()
        self.play_obj = None

        self.beep = pydub.generators.Sine(1200).to_audio_segment(duration=30).apply_gain(-30).raw_data
        pass

    def play(self, start_sec, end_sec, use_beep = True):
        self.stop()

        assert start_sec < end_sec
        sample_start = int(start_sec * 16_000)
        sample_end = int(end_sec * 16_000)
        wave_clip = self.wave_bytes[sample_start*2:sample_end*2]
        if use_beep:
            wave_clip += self.beep

        self.play_obj = simpleaudio.play_buffer(wave_clip, 1, 2, 16_000)

    def stop(self):
        if self.play_obj:
            self.play_obj.stop()
            self.play_obj = None

    def forward(self, sec:float):
        ...

    @staticmethod
    def load_audio(media_input: str, *, sr: int = 16000):
        FFMPEG = 'ffmpeg -nostdin -loglevel warning -threads 0 -y'
        try:
            cmd = f"{FFMPEG} -i {media_input} -f s16le -ac 1 -acodec pcm_s16le -ar {sr} -"
            out = subprocess.run(cmd.split(), capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


import pyaudio

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

    def play(self, start_sec:float, end_sec:float):
        asyncio.create_task(self.aplay(start_sec, end_sec))

    async def aplay(self, start_sec:float, end_sec:float):

        self.n_played += 1
        try:
            # wait not playing...
            while self.n_played >= 2:
                self.stop_play = True
                await asyncio.sleep(0.1)

            if self.play_to_end:
                end_sec = 99999

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
                self.audio.play(ssec, esec, use_beep= sec_type != 's.')

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

    clrs = [
        0xFF5733, 0x33FF57, 0x3357FF, 0xFF33A1, 0xFFC300,
        0x8D33FF, 0x33FFF5, 0xFF8D33, 0x57FF33, 0xA1FF33,
    ]
    clr_index = 0
    def draw_text(self, text, *, clr_index=None):
        if clr_index is None:
            clr_index = VlcPlayer.clr_index
            VlcPlayer.clr_index += 1

        clr_index = clr_index % len(VlcPlayer.clrs)
        self._draw_text(text, clr_argb=VlcPlayer.clrs[clr_index])




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
    def __init__(self, video_path = None) -> None:
        if DebugDiarUI._player:
            DebugDiarUI._player.clear()

        self.speaker = None
        self.player = DebugDiarUI._player = VlcPlayer()
        if video_path:
            self.set_videofile(video_path)

        self.prev_end = 0

    def set_videofile(self, video_path):
        assert video_path and Path(video_path).exists()
        self.player.set_file(video_path)
        self.video_path = video_path


    @singledispatchmethod
    def set_segment(self, anno:Annotation, *, min_sec = 0):
        self.rawsegs = []
        for turn, _, speaker_tag in anno.itertracks(yield_label=True):
            if turn.end - turn.start > min_sec:
                self.rawsegs.append( Speaker(turn.start, turn.end, speaker_tag) )
        self.anno = anno
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
                await asyncio.sleep(1.5) # interval between segments

        except AttributeError as ex:
            print('aplay_all.ex:', ex)
            pass
        print('1. aplay_all done', self.is_playall)
        self.is_playall = False

    def on_play_all(self, btn, segs, slider:widgets.IntSlider):
        self.is_playall = not self.is_playall
        if self.is_playall:
            btn.description = 'Stop playing'
            return asyncio.create_task(self.aplay_all(segs, slider))
        else:
            btn.description = 'PlayAll'
            return

    def _play(self, tag, start, end):
        clr_idx = 0
        if m := re.search(r'\d+', tag):
            clr_idx = int(m.group()) + 1

        if self.speaker != 'OVERLAPPED' and tag == 'OVERLAPPED':
            # skipped
            return

        self.player.play(start, end)
        self.player.draw_text(f"{tag}\ndur={round(end-start,3)} sec, after={round(start-self.prev_end,3)}", clr_index=clr_idx)
        # if tag != 'INTER':
        self.prev_end = end

    def _interact_video(self, segs, label='', ui=None):

        count = len(segs)
        details = widgets.Label(value=f'')

        def fn_slider(idx):
            # self.is_playall = False
            seg = segs[idx]
            range = f"{to_hhmmss(seg.start_sec)} ~ {to_hhmmss(seg.end_sec)} / {seg.start_sec:.3f} ~ {seg.end_sec:.3f}"
            details.value = f"Range= {range}     [{seg.speaker_tag}]"

            # if self.vlc: self.vlc.stop()
            self._play(seg.speaker_tag, seg.start_sec, seg.end_sec)

        title = widgets.Label(value=f'> {label}: {count= }',
                            style={'background': 'green', 'text_color': 'white'},
                            layout=widgets.Layout(width='400px'),)

        # [0, max]
        slider = widgets.IntSlider(
            description=f'clips: ',
            min=0, max= max(0, len(segs)-1), step=1,
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            )
        slider.observe(lambda x: fn_slider(x['new']), names='value', type='change')
        def slider_value(val:int):
            if 0<= val < count: slider.value = val

        btn_prev = widgets.Button(description='<<')
        btn_next = widgets.Button(description='>>')

        replay_btn = widgets.Button(description='Replay', )
        playall_btn = widgets.Button(description='PlayAll', )
        replay_btn.on_click(lambda btn: fn_slider(slider.value))
        playall_btn.on_click(lambda btn: self.on_play_all(btn, segs, slider))
        btn_prev.on_click(lambda b: slider_value(slider.value-1))
        btn_next.on_click(lambda b: slider_value(slider.value+1))
        if ui: fn_slider(0) # play initial

        vbox = widgets.VBox([title, details, slider, widgets.HBox([replay_btn, playall_btn, btn_prev, btn_next])])
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
                update_display( self.anno.label_support(speaker) , display_id=self.disp_id)

        speakers = set( seg.speaker_tag for seg in self.rawsegs)
        speakers = [SPEAKER_ALL, *sorted(speakers)]
        dropdown = widgets.Dropdown(options=speakers, description='Speaker: ')
        dropdown.observe(select_speaker, names='value')
        display(widgets.HBox([dropdown, lbl_title]))
        select_speaker({"new": SPEAKER_ALL})

        btn_ff5 = widgets.Button(description='+ 5sec')
        btn_ff10 = widgets.Button(description='+ 10sec')
        btn_ff5.on_click(lambda b: (self.player.forward(5)))
        btn_ff10.on_click(lambda b: (self.player.forward(10)))
        display(widgets.HBox([btn_ff5, btn_ff10]))

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

        def on_checkbox_change(change):
            desc, value = change['owner'].description, change['new']
            if desc == 'Play to end':
                self.player.play_to_end = value
            elif desc == 'Play start/end only':
                self.player.play_boundary = value

        cb_play_boundary.observe(on_checkbox_change, names='value')
        cb_play_cont.observe(on_checkbox_change, names='value')

        btn_close = widgets.Button(description='Close')
        btn_close.on_click(lambda b: (self.player.clear()))
        display(widgets.HBox([cb_play_boundary, cb_play_cont]), btn_close)
        pass

# dui = DebugUI(mp4file)
# dui.set_segment(no_shortvoice_segs)