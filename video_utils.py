

import vlc
import time

import traceback

class VlcPlayer:
    def __init__(self):
        opts = '--video-on-top --no-sub-autodetect-file'
        self.inst = vlc.Instance(opts.split())
        self.player = self.inst.media_player_new()

    def set_media(self, media_path:str):
        self.player.stop()
        self.player.set_media( self.inst.media_new(media_path) )

    def _play(self, start_ms:int):
        if self.player.get_state() != vlc.State.Playing:
            self.player.play()

        for _ in range(20):
            time.sleep(0.1)
            if self.player.get_state() == vlc.State.Playing:
                self.player.set_time(start_ms)  # Jump to start time
                return
        raise RuntimeError(f"_play timeout: invalid state: {self.player.get_state()}")

    def set_text(self, text:str, clr_argb:int=0xFF5733):
        self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Enable, 1)
        self.player.video_set_marquee_string(vlc.VideoMarqueeOption.Text, text)
        self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Position, 0) # vlc.Position.bottom_right)
        self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Color, clr_argb|0xff000000)  # Red
        self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Opacity, 255)
        self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Timeout, 0)
        self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Size, 60)

    def play(self, start_sec:float, end_sec:float, text:str=""):
        start_ms = int(start_sec*1000)
        end_ms = int(end_sec*1000)
        self._play(start_ms)
        self.set_text(text)

        while self.player.get_time() < end_ms:
            time.sleep(0.1)  # Keep playing until end time is reached

    def clear(self):
        try:
            self.player.video_set_marquee_int(vlc.VideoMarqueeOption.Enable, 0)
            self.player.stop()
            self.player.release()
            self.inst.release()
            print("vlcp cleared...")
        except:
            pass

    # context manager
    def __enter__(self):
        # No additional setup needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        if exc_type:
            exstr = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
            exstr = exstr.replace('File "/', 'File /')
            exstr = exstr.replace('", line ', ':')
            print('vlc error:', exstr)

