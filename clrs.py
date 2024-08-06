import re

class text_color:
    black,red,green,yellow,blue,magenta,cyan,white, gray = [*range(30,38), 90] # fgclr,  [*range(90,98), ''] # light-fgclr
    light = 60
    bold, underline, strike  = 1,4,9 # attrs supported on vscode notebook.
    bold, italic, underline, blink_slow, blink_fast, inverse, hidden, strike = 1, 3, 4, 5, 6, 7, 8, 9  # attrs supported on vscode notebook.

    def __init__(self, fg:int=0,bg:int=0,attr:int=0):
        self.clr = f'\33[{attr}'
        # assert fg != bg, f"invalid {fg=}, {bg=}"
        if fg: self.clr += f';{fg}'
        if bg: self.clr += f';{bg+10}'
        self.clr += 'm'

    def __ror__(self, obj): return self.clr + str(obj) + '\33[0m'

    @staticmethod
    def decolorize(text):
        return re.sub(r"\x1b\[[\d;]+m", "", text)



# https://stackabuse.com/how-to-print-colored-text-in-python/
def clr256(fg,bg):
    fgclr = f"\033[38;5;{fg}m" if fg>=0 else ''
    bgclr = f"\033[48;5;{bg}m" if bg>=0 and fg != bg else ''
    return fgclr+bgclr

def colors_256(fg, bg=-1):
    clr = clr256(fg,bg)
    reset = '\033[0m'
    clrtext = clr.replace('\033', '\\033'); clrtext = f"{clrtext:<28}"
    clrtext = f" ({fg}, {bg})"; clrtext = f"{clrtext:<12}"
    return clr + clrtext + reset + ('\n' if (fg+1) % 8 == 0 else '')

def colors():
    bg = 236
    print("\nThe 256 colors scheme is:")
    print(''.join([colors_256(x, bg) for x in range(256)]))


#
#
#
red = text_color(text_color.red)
green = text_color(text_color.green)
blue = text_color(text_color.blue)
yellow = text_color(text_color.yellow)
magenta = text_color(text_color.magenta)
cyan = text_color(text_color.cyan)
gray = text_color(text_color.gray)

if __name__ == '__main__':
    colors()