import re

class text_color:
    black,red,green,yellow,blue,magenta,cyan,white,gray = [*range(30,38), 90] # fgclr,  [*range(90,98), ''] # light-fgclr
    bold, italic, underline, strike = 1, 3, 4, 9  # attrs supported on vscode notebook.
    def __init__(self, fg,bg=0,attr=0):
        attr = f'{attr};' if attr > 0 else ''
        bg = f'{bg+10};' if bg > 0 else ''
        self.clr = f'\33[{attr}{bg}{fg}m'
    def __ror__(self, obj): return self.clr + str(obj) + '\33[0m'
    @staticmethod
    def all(): return (text_color(clr) for clr in [*range(30,38), 90])
# black,red,green,yellow,blue,magenta,cyan,white,gray = text_color.all()

black,red,green,yellow,blue,magenta,cyan,white,gray = text_color.all()

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

if __name__ == '__main__':
    colors()