
# !pip install python-vlc
import asyncio, os
import vlc

class VlcPlayer:
    def __init__(self, width = 1920*2, height = 1080*2):
        os.environ["VLC_VERBOSE"] = str("-1")

        opts = []
        opts.extend("--video-on-top --no-sub-autodetect-file".split())
        # opts.extend(f"--video-on-top --width={width} --height={height}".split())

        self.instance = vlc.Instance(opts)
        self.player = self.instance.media_player_new()
        self.player.video_set_scale(1.5)
        self.stop_play = True
        self.async_play_task = None

    def set_file(self, mp4path):
        media = self.instance.media_new(mp4path)
        self.player.set_media(media)

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

    async def _aplay(self, start_sec:float, end_sec:float):

        # print(f'play: {start_sec:.1f} ~ {end_sec:.1f}')
        self.player.play()

        while self.player.get_state() != vlc.State.Playing:
            await asyncio.sleep(0.1)

        self.player.set_time(int(start_sec * 1000))
        self.end_ms = int(end_sec * 1000)

        while self.stop_play == False and self.player.get_time() < self.end_ms:
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




from pathlib import Path
from jjutils.diar_utils import (
    get_inter_pklsegments,
    PklSegment,
)

from IPython.display import display
import ipywidgets as widgets
import time

class DebugUI:

    def __init__(self, video_path = None) -> None:
        self.player = VlcPlayer()
        if video_path:
            self.set_videofile(video_path)

    def set_videofile(self, video_path):
        assert video_path and Path(video_path).exists()
        self.player.set_file(video_path)

    def set_segment(self, rawsegs:list[PklSegment], *, crop_start=0, crop_end=0, min_sec = 0):

        if crop_end > 0:
            rawsegs = [seg for seg in rawsegs if crop_start <= seg.start_sec and seg.end_sec < crop_end]

        self.rawsegs = [i for i in rawsegs if i.end_sec - i.start_sec > min_sec]
        self.segs_inter = get_inter_pklsegments(self.rawsegs)

        self._setup_ui()

    def _setup_ui(self):
        self.roi_widgets = None
        def select_speaker(change):
            speaker = change['new']

            segs = self.rawsegs if speaker == 'All' else [item for item in self.rawsegs if item.speaker_tag == speaker]
            self.roi_widgets = self._interact_video(segs, f'diar: {speaker}', ui=self.roi_widgets)

        speakers = set( seg.speaker_tag for seg in self.rawsegs)
        speakers = ['All', *sorted(speakers)]
        dropdown = widgets.Dropdown(options=speakers, description='Speaker: ')
        dropdown.observe(select_speaker, names='value')
        display(dropdown)
        select_speaker({"new":"All"})

        btn_ff = widgets.Button(description='>>')
        btn_ff.on_click(lambda b: (self.player.forward(5000)))
        display(btn_ff)

        self._interact_video(self.segs_inter, f'diar: inter')

        btn_close = widgets.Button(description='Close')
        btn_close.on_click(lambda b: (self.player.clear()))
        display(btn_close)
        pass

    def _interact_video(self, pklsegs, label='', ui=None):

        count = len(pklsegs)
        details = widgets.Label(value=f'')

        def fn_slider(idx):
            pklseg = pklsegs[idx]
            details.value = f"{pklseg}"

            # if self.vlc: self.vlc.stop()
            dur = round(pklseg.end_sec - pklseg.start_sec,1)
            self.player.play(pklseg.start_sec, pklseg.end_sec)
            self.player.draw_text(f"{pklseg.speaker_tag}, {dur:.1f} sec", timeout=int(dur*1000))

        replay_btn = widgets.Button(description='Replay', )

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
        replay_btn.on_click(lambda btn: fn_slider(slider.value))

        vbox = widgets.VBox([title, details, slider, replay_btn])
        if ui == None:
            ui = vbox
            display(ui, display_id='11')

        ui.children = vbox.children
        return ui

# dui = DebugUI(mp4file)
# dui.set_segment(no_shortvoice_segs)