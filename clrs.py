import re

class text_color:
    black,red,green,yellow,blue,magenta,cyan,white,gray = [*range(30,38), 90] # fgclr,  [*range(90,98), ''] # light-fgclr
    bold, italic, underline, strike = 1, 3, 4, 9  # attrs supported on vscode notebook.
    def __init__(self, fg:int=0,bg:int=0,attr:int=0):
        self.clr = f'\33[{attr}'
        # assert fg != bg, f"invalid {fg=}, {bg=}"
        if fg: self.clr += f';{fg}'
        if bg: self.clr += f';{bg+10}'
        self.clr += 'm'

    def __ror__(self, obj):
        if self.tmpclr:
            text = self.tmpclr + str(obj) + '\33[0m'
            self.tmpclr = None
            return text
        return self.clr + str(obj) + '\33[0m'

    def __call__(self,*,fg=0, bg=0):
        _bg,_fg = self.clr[2:-1].split(';')
        if bg>0: _bg = bg+10
        if fg>0: _fg = fg
        self.tmpclr = f"\33[{_bg};{_fg}m"
        return self
    @staticmethod
    def all(): return (text_color(clr) for clr in [*range(30,38), 90])


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