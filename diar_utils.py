

import re
import pickle
from dataclasses import dataclass, asdict
from jjutils.audio_utils import hhmmss_to_seconds
from jjutils.audio_utils import to_hhmmss

@dataclass
class PklSegment:
    start_sec: float = 0
    end_sec: float = 0
    speaker_tag: int = None

    def copy(self):
        return PklSegment(self.start_sec, self.end_sec, self.speaker_tag)




# list[PklSegment] to Annotation.
def pklsegments_to_annotation(pklsegs:list[PklSegment]) -> object:
    from pyannote.core import Annotation, Segment
    anno = Annotation()
    for seg in pklsegs:
        anno[Segment(seg.start_sec, seg.end_sec)] = seg.speaker_tag
    return anno


# remove_overlapped
def remove_overlapped(pklsegs:list[PklSegment], debug=False) -> list[PklSegment]:

    old_seg = PklSegment()
    no_overlapped_segs = []

    for seg in pklsegs:
        duration = seg.end_sec - seg.start_sec

        # 2.1 tweak: overlapped
        # if segment.start_sec+1.5 < old_seg.end_sec:
        if seg.start_sec < old_seg.end_sec:
            assert old_seg.speaker_tag != seg.speaker_tag
            dur_old = old_seg.end_sec - old_seg.start_sec

            # if current segment is short, skip it.
            if (dur_old * 1.1) > duration and duration < 2.0:
                if debug:
                    print(
                        f'skipped current: ({to_hhmmss(seg.start_sec, True)} ~ {to_hhmmss(seg.end_sec, True)}, {duration:.1f}):{seg.speaker_tag}'
                        f'    old: {to_hhmmss(old_seg.start_sec, True)} ~ {to_hhmmss(old_seg.end_sec, True)}, {dur_old:.1f}):{old_seg.speaker_tag}'
                        )
                continue

            # update old segment
            dur_old_no_overlapped = seg.start_sec - old_seg.start_sec
            assert dur_old_no_overlapped < dur_old
            if dur_old_no_overlapped > 1.0: # if longer than 1sec
                if debug:
                    print(
                        f'update old: {to_hhmmss(old_seg.start_seg, True)} ~ {to_hhmmss(old_seg.end_seg, True)}, {dur_old:.1f}) ->'
                        f' {dur_old_no_overlapped:.1f}):{old_seg.speaker_tag}'
                        )
                no_overlapped_segs[-1] = PklSegment(old_seg.start_sec, seg.start_sec, old_seg.speaker_tag)
            else:
                no_overlapped_segs.pop()


        no_overlapped_segs.append( seg.copy() )
        old_seg = seg

    return no_overlapped_segs





# remove_short_mute
# - merge short segment in long speaking into long speaking
def remove_short_mute(pklsegs:list[PklSegment], debug=False) -> list[PklSegment]:
    merged_segs = []
    old_seg = PklSegment()

    MUTE_THRESHOLD = 1.0
    SHORT_NOISE_THRESHOLD = 0.5

    if debug:
        print(f"{MUTE_THRESHOLD=}")
        print(f"{SHORT_NOISE_THRESHOLD=}")

    pos = 0
    while pos < len(pklsegs):
        segment = pklsegs[pos]
        pos += 1

        if pos < len(pklsegs) and old_seg.speaker_tag != segment.speaker_tag:
            # check current segment is noise or not.
            # |--- A --- B? --A--|, Assume B as noise.
            dur_old = old_seg.end_sec - old_seg.start_sec
            duration = segment.end_sec - segment.start_sec
            next_segment = pklsegs[pos]
            if duration < SHORT_NOISE_THRESHOLD and next_segment.speaker_tag == old_seg.speaker_tag:
                if debug:
                    print(
                        f'noise: ({to_hhmmss(segment.start_sec, True)}, {duration=:.1f}) : convert {segment.speaker_tag} -> {old_seg.speaker_tag}'
                        )
                segment.speaker_tag = old_seg.speaker_tag

        if old_seg.speaker_tag == segment.speaker_tag:
            dur_mute = segment.start_sec - old_seg.end_sec
            if dur_mute < MUTE_THRESHOLD:
                # merge with old seg
                segment = PklSegment(old_seg.start_sec, segment.end_sec, segment.speaker_tag)
                # if debug:
                #     duration = segment.end_sec - segment.start_sec
                #     dur_old = old_seg.end_sec - old_seg.start_sec
                #     print(
                #         f'merged: ({to_hhmmss(old_seg.start_sec, True)} ~ {to_hhmmss(old_seg.end_sec, True)}, {dur_old:.1f})'
                #         f' -> {to_hhmmss(segment.start_sec, True)} ~ {to_hhmmss(segment.end_sec, True)}, {duration:.1f}) :{old_seg.speaker_tag}'
                #         )
                merged_segs.pop() # remove old


        merged_segs.append(segment.copy())
        old_seg = segment

    return merged_segs






from jjutils.audio_utils import to_hhmmss
# 1
# 00:00:00,000 --> 00:00:29,980
# ご視聴ありがとうございました

# 2
def pklsegments_to_srt(pklsegs:list[PklSegment], srt_path):
    with open(srt_path, "w") as fpout:
        nth = 1
        pklseg = PklSegment(0,0,None)
        for segment in pklsegs:
            (start, end, speaker_tag) = segment.start_sec, segment.end_sec, segment.speaker_tag

            if pklseg.end_sec < start: # blank
                print(f"{nth}\n{to_hhmmss(pklseg.end_sec)} --> {to_hhmmss(start-0.1)}\n---\n\n", file=fpout)
                nth += 1
            print(f"{nth}\n{to_hhmmss(start)} --> {to_hhmmss(end)}\n{speaker_tag}\n\n", file=fpout)
            nth += 1
            pklseg.start_sec = start
            pklseg.end_sec = end
        print('done', srt_path)



# srt to list[PklSegment]

def srt_to_pklsegments(srt_path):
    with open(srt_path, encoding='utf-8') as fp:
        matches = re.findall(r'\d+\n(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n([^\n]+)\n', fp.read(), re.M|re.DOTALL)
        print( len(matches) )

    pklsegs = []
    for (s,e,tag) in matches:
        if not tag.isdigit(): continue
        pklsegs.append(PklSegment(hhmmss_to_seconds(s), hhmmss_to_seconds(e), int(tag)))
    return pklsegs








def build_m3u(pklsegs:list[PklSegment], m3ufile:str, mp4file:str, speaker_map = dict()):
    BEGIN_MARGIN = 0.0
    n_item = 0
    old_range = PklSegment()
    m3u_segments = []
    with open(m3ufile, 'w') as fp:
        fp.write('#EXTM3U\n')

        for segment in pklsegs:
            (ssec, esec) = segment.start_sec, segment.end_sec
            speaker_id = segment.speaker_tag
            duration = esec - ssec

            # 2.1 tweak
            if ssec+1.5 < old_range.end_sec:
                print(
                    f'debug: dup({to_hhmmss(ssec, True)} ~ {to_hhmmss(esec, True)}, {duration:.1f}):{speaker_id}'
                    f'    old: {to_hhmmss(old_range.start_sec, True)} ~ {to_hhmmss(old_range.end_sec, True)}, {old_range.end_sec-old_range.start_sec:.1f}):{old_range.speaker_tag}'
                    )
                continue

            dur_mute = ssec - old_range.end_sec

            if dur_mute > 0.5:
                n_item += 1
                mute_end = ssec - 0.5
                fp.write(
                    f'#EXTINF:{int(dur_mute)}, NoVoice - {n_item:03d}, {to_hhmmss(old_range.end_sec, True)} ~ {to_hhmmss(mute_end, True)} ({old_range.end_sec:.1f}~{ssec:.1f}, {dur_mute:.1f})\n'
                    f'#EXTVLCOPT:start-time={old_range.end_sec}\n'
                    f'#EXTVLCOPT:stop-time={ssec}\n'
                    f'{mp4file}\n'
                )

            n_item += 1
            ssec -= BEGIN_MARGIN
            fp.write(
                "\n"
                f'#EXTINF:{int(duration)}, {speaker_id} - {n_item:03d}, {to_hhmmss(ssec, True)} ~ {to_hhmmss(esec, True)} ({ssec:.1f}~{esec:.1f}, {duration:.1f})\n'
                f'#EXTVLCOPT:start-time={ssec}\n'
                f'#EXTVLCOPT:stop-time={esec}\n'
                f'{mp4file}\n'
                #vlc://pause:1
            )
            old_range = PklSegment(ssec, esec, speaker_id)
    print('gen:', m3ufile)





def segments_to_pickle(segments:list[PklSegment], pickle_file:str):
    with open(pickle_file, 'wb') as fp:
        pickle.dump([asdict(seg) for seg in segments], fp)

