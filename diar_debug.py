import asyncio, os
import vlc # !pip install python-vlc
import gc, time
import os

class VlcPlayer:
    def __init__(self, width = 1920*2, height = 1080*2):
        os.environ["VLC_VERBOSE"] = str("-1")

        opts = []
        opts.extend("--video-on-top --no-sub-autodetect-file".split())
        # opts.extend(f"--video-on-top --width={width} --height={height}".split())

        self.instance = vlc.Instance(opts)
        self.vlcp = self.instance.media_player_new()
        # self.player.video_set_scale(3)
        self.stop_play = True
        self.async_play_task = None
        self.text = ''
        self.stop_ms = 0
        self.play_boundary = False

    def set_file(self, mp4path):
        media = self.instance.media_new(mp4path)
        self.vlcp.set_media(media)

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
        # if self.async_play_task:
        #     del self.async_play_task
            # self.stop_play = True
            # while self.async_play_task != None: time.sleep(0.1)

        del self.vlcp
        del self.instance
        self.instance = self.vlcp = None
        gc.collect()

    async def _aplay(self, start_sec:float, end_sec:float):
        # print(f'play: {start_sec:.1f} ~ {end_sec:.1f}')
        self.vlcp.play()

        while self.vlcp.get_state() != vlc.State.Playing:
            await asyncio.sleep(0.1)

        self.vlcp.set_time(int(start_sec * 1000))
        self.end_ms = int(end_sec * 1000)

        while self.stop_play == False:
            remained = self.end_ms - self.vlcp.get_time()
            if remained <= 0: break
            self.vlcp.video_set_marquee_string(vlc.VideoMarqueeOption.Text,
                                                 self.text + f'\n: {int(remained/1000)} sec remained')
            await asyncio.sleep(0.1)


        self.vlcp.pause()
        self.async_play_task = None

    async def aplay_old(self, start_sec:float, end_sec:float):
        if self.async_play_task and not self.async_play_task.done():
            self.stop_play = True
            await self.async_play_task

        self.stop_play = False
        self.async_play_task = asyncio.create_task(self._aplay(start_sec, end_sec))

    def play(self, start_sec:float, end_sec:float):
        asyncio.create_task(self.aplay(start_sec, end_sec))

    async def aplay(self, start_sec:float, end_sec:float):

        try:
            # wait not playing...
            self.stop_play = True
            while self.vlcp.get_state() == vlc.State.Playing:
                await asyncio.sleep(0.2)

            # wait playing...
            self.vlcp.play()
            while self.vlcp.get_state() != vlc.State.Playing:
                await asyncio.sleep(0.1)

            duration = end_sec - start_sec
            if duration - 5*2 > 3 and self.play_boundary:
                ranges = [(start_sec, start_sec+5, 's.'), (end_sec-5, end_sec, 'e.')]
            else:
                ranges = [(start_sec, end_sec, '')]

            self.stop_play = False
            for start_sec, end_sec, sec_type in ranges:
                self.vlcp.set_time(int(start_sec * 1000))
                self.end_ms = int(end_sec * 1000)

                while self.stop_play == False:
                    remained = self.end_ms - self.vlcp.get_time()
                    if remained <= 0: break
                    self.vlcp.video_set_marquee_string(vlc.VideoMarqueeOption.Text,
                                                        self.text + f'\n: {sec_type}{int(remained/1000)} sec remained')
                    await asyncio.sleep(0.1) # update text.
            # for
        except AttributeError:
            pass

        if self.vlcp:
            self.vlcp.pause()


    def is_playing(self):
        return self.vlcp and self.vlcp.get_state() == vlc.State.Playing # and self.vlcp.get_time() < self.end_ms

    def forward(self, ms=1000):
        if self.vlcp.get_state() == vlc.State.Playing:
            curpos = self.vlcp.get_time()
            self.vlcp.set_time(curpos + ms)

    def _draw_text(self, text, *, timeout = 2000, position=0, clr_argb):
        player = self.vlcp
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Enable, 1)

        player.video_set_marquee_string(vlc.VideoMarqueeOption.Text, text)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Position, position)  # Top-left
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Color, clr_argb|0xff000000)  # Red
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Opacity, 255)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Timeout, timeout)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Size, 40)
        self.text = text

    clrs = [
        0xFF5733, 0x33FF57, 0x3357FF, 0xFF33A1, 0xFFC300,
        0x8D33FF, 0x33FFF5, 0xFF8D33, 0x57FF33, 0xA1FF33,
    ]
    clr_index = 0
    def draw_text(self, text, timeout = 2000, position=0):
        self._draw_text(text, timeout=timeout,
                       position=position,
                       clr_argb=VlcPlayer.clrs[VlcPlayer.clr_index],
                       )
        VlcPlayer.clr_index = (VlcPlayer.clr_index+1) % len(VlcPlayer.clrs)



from pathlib import Path
from jjutils.diar_utils import (
    get_inter_pklsegments,
    PklSegment,
)

from pyannote.database.util import load_rttm

from IPython.display import display
import ipywidgets as widgets
from functools import singledispatch, singledispatchmethod
from jjutils.diar_utils import (
    annotation_to_pklsegments
)

from pyannote.core import Annotation, Segment

class DebugUI:
    _player = None
    def __init__(self, video_path = None) -> None:
        if DebugUI._player:
            DebugUI._player.clear()

        self.player = DebugUI._player = VlcPlayer()
        if video_path:
            self.set_videofile(video_path)

    def set_videofile(self, video_path):
        assert video_path and Path(video_path).exists()
        self.player.set_file(video_path)
        self.video_path = video_path

    @singledispatchmethod
    def set_segment(self, rawsegs:list[PklSegment], *, min_sec = 0):

        self.rawsegs = [i for i in rawsegs if i.end_sec - i.start_sec > min_sec]
        # self.segs_inter = get_inter_pklsegments(self.rawsegs)

        self._setup_ui()

    @set_segment.register
    def _(self, anno_path:str, *, min_sec = 0):

        if anno_path.endswith('.rttm'):
            _, anno = load_rttm(anno_path).popitem()
            segs = annotation_to_pklsegments(anno)
            self.set_segment(segs, min_sec=min_sec)
        else:
            assert False, f"invalid file format: {anno_path}"

    @set_segment.register
    def _(self, anno:Annotation, *, min_sec = 0):
        segs = annotation_to_pklsegments(anno)
        self.set_segment(segs, min_sec=min_sec)

    def _setup_ui(self):
        self.roi_widgets = None
        self.is_playall = False


        lbl_title = widgets.Label(value= f"[ {self.video_path} ]")
        lbl_title.layout = widgets.Layout(margin='0 0 0 20px')
        def select_speaker(change):
            speaker = change['new']

            segs = self.rawsegs if speaker == 'All' else [item for item in self.rawsegs if item.speaker_tag == speaker]
            self.roi_widgets = self._interact_video(segs, f'diar: {speaker}', ui=self.roi_widgets)
            # slider = [w for w in self.roi_widgets.children if isinstance(w, widgets.IntSlider)]

        speakers = set( seg.speaker_tag for seg in self.rawsegs)
        speakers = ['All', *sorted(speakers)]
        dropdown = widgets.Dropdown(options=speakers, description='Speaker: ')
        dropdown.observe(select_speaker, names='value')
        display(widgets.HBox([dropdown, lbl_title]))
        select_speaker({"new":"All"})

        btn_ff5 = widgets.Button(description='+ 5sec')
        btn_ff10 = widgets.Button(description='+ 10sec')
        btn_ff5.on_click(lambda b: (self.player.forward(5_000)))
        btn_ff10.on_click(lambda b: (self.player.forward(10_000)))
        display(widgets.HBox([btn_ff5, btn_ff10]))

        # self._interact_video(self.segs_inter, f'diar: inter')

        cb_play_boundary = widgets.Checkbox(
            value=False,
            description='Play start/end only',
            indent=False
        )

        def on_checkbox_change(change):
            self.player.play_boundary = change['new']
        cb_play_boundary.observe(on_checkbox_change, names='value')

        btn_close = widgets.Button(description='Close')
        btn_close.on_click(lambda b: (self.player.clear()))
        display(cb_play_boundary, btn_close)
        pass

    async def aplay_all(self, pklsegs, slider:widgets.IntSlider):
        print('0. aplay_all done', self.is_playall)
        try:
            pklseg = pklsegs[slider.value]
            dur = round(pklseg.end_sec - pklseg.start_sec,1)
            self.player.play(pklseg.start_sec, pklseg.end_sec)
            self.player.draw_text(f"{pklseg.speaker_tag}, {dur:.1f} sec", timeout=int(dur*1000))
            while not self.player.is_playing(): await asyncio.sleep(0.5)
            while self.player.is_playing(): await asyncio.sleep(0.5)

            while slider.value < slider.max and self.is_playall:
                slider.value = slider.value + 1
                while not self.player.is_playing():
                    await asyncio.sleep(0.5)

                while self.player.is_playing():
                    await asyncio.sleep(0.5)

                await asyncio.sleep(1.5)


            # for seg in pklsegs:
            #     if not self.is_playall: break
            #     dur = round(seg.end_sec - seg.start_sec,1)
            #     self.player.play(seg.start_sec, seg.end_sec)
            #     self.player.draw_text(f"{seg.speaker_tag}, {dur:.1f} sec", timeout=int(dur*1000))
            #     await self.player.aplay(seg.start_sec, seg.end_sec)
            #     await asyncio.sleep(2)
        except AttributeError:
            pass
        print('1. aplay_all done', self.is_playall)
        self.is_playall = False

    def on_play_all(self, btn, pklsegs, slider:widgets.IntSlider):
        self.is_playall = not self.is_playall
        if self.is_playall:
            btn.description = 'Stop playing'
            return asyncio.create_task(self.aplay_all(pklsegs, slider))
        else:
            btn.description = 'PlayAll'
            return

    def _interact_video(self, pklsegs, label='', ui=None):

        count = len(pklsegs)
        details = widgets.Label(value=f'')

        def fn_slider(idx):
            # self.is_playall = False
            pklseg = pklsegs[idx]
            details.value = f"{pklseg}"

            # if self.vlc: self.vlc.stop()
            dur = round(pklseg.end_sec - pklseg.start_sec,1)
            self.player.play(pklseg.start_sec, pklseg.end_sec)
            self.player.draw_text(f"{pklseg.speaker_tag}, {dur:.1f} sec", timeout=int(dur*1000))

        title = widgets.Label(value=f'> {label}: {count= }',
                            style={'background': 'green', 'text_color': 'white'},
                            layout=widgets.Layout(width='400px'),)

        # [0, max]
        slider = widgets.IntSlider(
            description=f'clips: ',
            min=0, max= max(0, len(pklsegs)-1), step=1,
            continuous_update=False,
            style={'description_width': 'initial'},
            )
        slider.observe(lambda x: fn_slider(x['new']), names='value', type='change')
        def slider_value(val:int):
            if 0<= val < count: slider.value = val

        btn_prev = widgets.Button(description='<<')
        btn_next = widgets.Button(description='>>')

        replay_btn = widgets.Button(description='Replay', )
        playall_btn = widgets.Button(description='PlayAll', )
        replay_btn.on_click(lambda btn: fn_slider(slider.value))
        playall_btn.on_click(lambda btn: self.on_play_all(btn, pklsegs, slider))
        btn_prev.on_click(lambda b: slider_value(slider.value-1))
        btn_next.on_click(lambda b: slider_value(slider.value+1))
        if ui: fn_slider(0) # play initial

        vbox = widgets.VBox([title, details, slider, widgets.HBox([replay_btn, playall_btn, btn_prev, btn_next])])
        if ui == None:
            ui = vbox
            display(ui, display_id='11')

        ui.children = vbox.children
        return ui

# dui = DebugUI(mp4file)
# dui.set_segment(no_shortvoice_segs)