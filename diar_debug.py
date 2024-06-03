import asyncio, os
import vlc # !pip install python-vlc
import gc

class VlcPlayer:
    def __init__(self, width = 1920*2, height = 1080*2):
        os.environ["VLC_VERBOSE"] = str("-1")

        opts = []
        opts.extend("--video-on-top --no-sub-autodetect-file".split())
        # opts.extend(f"--video-on-top --width={width} --height={height}".split())

        self.instance = vlc.Instance(opts)
        self.player = self.instance.media_player_new()
        # self.player.video_set_scale(3)
        self.stop_play = True
        self.async_play_task = None
        self.text = ''

    def set_file(self, mp4path):
        media = self.instance.media_new(mp4path)
        self.player.set_media(media)

        media.parse()
        width,height = self.player.video_get_size()
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
        self.player.stop()
        # if self.async_play_task:
        #     del self.async_play_task
            # self.stop_play = True
            # while self.async_play_task != None: time.sleep(0.1)

        del self.player
        del self.instance
        self.instance = self.player = None
        gc.collect()

    async def _aplay(self, start_sec:float, end_sec:float):

        # print(f'play: {start_sec:.1f} ~ {end_sec:.1f}')
        self.player.play()

        while self.player.get_state() != vlc.State.Playing:
            await asyncio.sleep(0.1)

        self.player.set_time(int(start_sec * 1000))
        self.end_ms = int(end_sec * 1000)

        while self.stop_play == False:
            remained = self.end_ms - self.player.get_time()
            if remained <= 0: break
            self.player.video_set_marquee_string(vlc.VideoMarqueeOption.Text,
                                                 self.text + f'\n: {int(remained/1000)} sec remained')
            await asyncio.sleep(0.1)


        self.player.pause()
        self.async_play_task = None

        # print('play-done')

    def play(self, start_sec:float, end_sec:float):
        return asyncio.create_task(self.aplay(start_sec, end_sec))

    async def aplay(self, start_sec:float, end_sec:float):
        self.start_sec = start_sec
        self.end_sec = end_sec

        if self.async_play_task and not self.async_play_task.done():
            self.stop_play = True
            await self.async_play_task

        self.stop_play = False
        self.async_play_task = asyncio.create_task(self._aplay(start_sec, end_sec))

    def is_playing(self):
        return self.player.get_time() < self.end_ms

    def forward(self, ms=1000):
        if self.player.get_state() == vlc.State.Playing:
            curpos = self.player.get_time()
            self.player.set_time(curpos + ms)

    def draw_text(self, text, timeout = 2000, clr_argb=0xffff0000, position=0):
        player = self.player
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Enable, 1)

        player.video_set_marquee_string(vlc.VideoMarqueeOption.Text, text)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Position, position)  # Top-left
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Color, clr_argb)  # Red
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Opacity, 255)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Timeout, timeout)
        player.video_set_marquee_int(vlc.VideoMarqueeOption.Size, 40)
        self.text = text




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


from IPython.display import Javascript
def set_focus(widget):
    display(Javascript(f"""
        var slider = document.querySelector("#{widget.model_id} input");
        if (slider) {{
            slider.focus();
        }}
    """))


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
    def set_segment(self, rawsegs:list[PklSegment], *, crop_start=0, crop_end=0, min_sec = 0):

        if crop_end > 0:
            rawsegs = [seg for seg in rawsegs if crop_start <= seg.start_sec and seg.end_sec < crop_end]

        self.rawsegs = [i for i in rawsegs if i.end_sec - i.start_sec > min_sec]
        self.segs_inter = get_inter_pklsegments(self.rawsegs)

        self._setup_ui()

    @set_segment.register
    def _(self, anno_path:str, *, crop_start=0, crop_end=0, min_sec = 0):

        if anno_path.endswith('.rttm'):
            _, anno = load_rttm(anno_path).popitem()
            segs = annotation_to_pklsegments(anno)
            self.set_segment(segs, crop_start=crop_start, crop_end=crop_end, min_sec=min_sec)
        else:
            assert False, f"invalid file format: {anno_path}"


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

        btn_ff = widgets.Button(description='>>')
        btn_ff.on_click(lambda b: (self.player.forward(5000)))
        display(btn_ff)

        self._interact_video(self.segs_inter, f'diar: inter')

        btn_close = widgets.Button(description='Close')
        btn_close.on_click(lambda b: (self.player.clear()))
        display(btn_close)
        pass

    async def aplay_all(self, pklsegs, slider:widgets.IntSlider):
        print('0. aplay_all done', self.is_playall)
        while slider.value < slider.max and self.is_playall:
            slider.value = slider.value + 1
            await asyncio.sleep(3)

        # for seg in pklsegs:
        #     if not self.is_playall: break
        #     dur = round(seg.end_sec - seg.start_sec,1)
        #     self.player.play(seg.start_sec, seg.end_sec)
        #     self.player.draw_text(f"{seg.speaker_tag}, {dur:.1f} sec", timeout=int(dur*1000))
        #     await self.player.aplay(seg.start_sec, seg.end_sec)
        #     await asyncio.sleep(2)
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

        slider = widgets.IntSlider(
            description=f'clips: ',
            min=0, max= max(0, len(pklsegs)-1), step=1,
            continuous_update=False,
            style={'description_width': 'initial'},
            )
        slider.observe(lambda x: fn_slider(x['new']), names='value', type='change')

        replay_btn = widgets.Button(description='Replay', )
        playall_btn = widgets.Button(description='PlayAll', )
        replay_btn.on_click(lambda btn: fn_slider(slider.value))
        playall_btn.on_click(lambda btn: self.on_play_all(btn, pklsegs, slider))
        if ui: fn_slider(0) # play initial

        vbox = widgets.VBox([title, details, slider, widgets.HBox([replay_btn, playall_btn])])
        if ui == None:
            ui = vbox
            display(ui, display_id='11')

        ui.children = vbox.children
        return ui

# dui = DebugUI(mp4file)
# dui.set_segment(no_shortvoice_segs)