import os
import select
import sys
import termios
import tty

def get_key(settings):
  tty.setraw(sys.stdin.fileno())
  rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
  if rlist:
    key = sys.stdin.read(1)
  else:
    key = None

  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
  return key
key_settings = termios.tcgetattr(sys.stdin)

while True:
    print(get_key(key_settings))
    if get_key(key_settings) is not None:
       break