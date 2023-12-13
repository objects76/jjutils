


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