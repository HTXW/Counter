"""
Created on Jun 18, 2012

@author: kermit

source from: http://code.activestate.com/recipes/134892/
"""


class _Getch:
    """Gets a single character from standard input.  Does not echo to the
    screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty
        import sys

    def __call__(self):
        import sys  # tty, termios
        # fd = sys.stdin.fileno()
        # old_settings = termios.tcgetattr(fd)
        try:
            # tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        except:
            raise
        # finally:
            # termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


if __name__ == '__main__':  # a little test
    print('Press a key')
    inkey = _Getch()
    import sys
    for i in range(sys.maxsize):
        k = inkey()
        if k == 'q':
            break
    print(('you pressed ', k))
