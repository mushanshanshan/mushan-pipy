'''
Date: 2022-07-20 15:22:19
LastEditors: mushan wwd137669793@gmail.com
LastEditTime: 2023-01-06 23:41:36
FilePath: /mushan/mushan/__init__.py
'''

from .chs.front_end import Frontend as ch_frontend
from .chs.ch_pho_sym import ch_symbols_dict

def ch_text_to_sequence(text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  text = text.split("-")
  for symbol in text:
    symbol_id = ch_symbols_dict[symbol]
    sequence += [symbol_id]
  return sequence