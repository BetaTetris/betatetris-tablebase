import os
import curses

COLOR_PAIRS_CACHE = {}


class TerminalColors(object):
    WHITE = '[0;37'
    CYAN = '[0;36'
    MAGENTA = '[0;35'
    BLUE = '[0;34'
    YELLOW = '[0;33'
    GREEN = '[0;32'
    RED = '[0;31'
    BLACK = '[0;30'
    END = '[0'


# Translates between the terminal notation of a color, to it's curses color number
TERMINAL_COLOR_TO_CURSES = {
    TerminalColors.BLACK: curses.COLOR_BLACK,
    TerminalColors.RED: curses.COLOR_RED,
    TerminalColors.GREEN: curses.COLOR_GREEN,
    TerminalColors.YELLOW: curses.COLOR_YELLOW,
    TerminalColors.BLUE: curses.COLOR_BLUE,
    TerminalColors.MAGENTA: curses.COLOR_MAGENTA,
    TerminalColors.CYAN: curses.COLOR_CYAN,
    TerminalColors.WHITE: curses.COLOR_WHITE
}


def _get_color(fg, bg):
    key = (fg, bg)
    if key not in COLOR_PAIRS_CACHE:
        # Use the pairs from 101 and after, so there's less chance they'll be overwritten by the user
        pair_num = len(COLOR_PAIRS_CACHE) + 1
        curses.init_pair(pair_num, fg, bg)
        COLOR_PAIRS_CACHE[key] = pair_num

    return COLOR_PAIRS_CACHE[key]


def _color_str_to_color_pair(color):
    if color == TerminalColors.END:
        fg = curses.COLOR_WHITE
    else:
        fg = TERMINAL_COLOR_TO_CURSES[color]
    color_pair = _get_color(fg, curses.COLOR_BLACK)
    return color_pair


def _add_line(y, x, window, line):
    # split but \033 which stands for a color change
    color_split = line.split('\033')

    # Print the first part of the line without color change
    default_color_pair = _get_color(curses.COLOR_WHITE, curses.COLOR_BLACK)
    window.addstr(y, x, color_split[0], curses.color_pair(default_color_pair))
    x += len(color_split[0])

    # Iterate over the rest of the line-parts and print them with their colors
    for substring in color_split[1:]:
        color_str = substring.split('m')[0]
        substring = substring[len(color_str)+1:]
        color_pair = _color_str_to_color_pair(color_str)
        window.addstr(y, x, substring, curses.color_pair(color_pair))
        x += len(substring)


def culour_addstr(window, y, x, string):
    assert curses.has_colors(), "Curses wasn't configured to support colors. Call curses.start_color()"
    for line in str(string).split(os.linesep):
        _add_line(y, x, window, line)
        y += 1

def reprint_line_bold(window, n: int) -> None:
    """
    Re-print the n-th (0-based) line in `window` in bold, preserving its
    existing color pair (and other non-bold attributes).
    """
    h, w = window.getmaxyx()
    if not (0 <= n < h):
        return

    width = max(0, w - 1)  # avoid writing into the newline column
    for x in range(width):
        cell = window.inch(n, x)
        ch = cell & curses.A_CHARTEXT
        attr = cell & curses.A_ATTRIBUTES

        # Preserve existing attributes & color, just force bold on.
        window.addch(n, x, ch, attr | curses.A_BOLD)

    window.refresh()
