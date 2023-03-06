from .cleaners import english_cleaners_to_phoneme, english_cleaners_wo_pho
from .symbols import symbol_to_id
import torch

class Frontend():
    def __init__(self):
        self.vocab_phones = symbol_to_id
        self.symbols_len = len(list(self.vocab_phones.keys()))
    
    def _intersperse(self, seq, item):
        result = [item] * (len(seq) * 2 + 1)
        result[1::2] = seq
        return result
    
    def textonly_to_idx(self, text, add_blank=False):
        text = english_cleaners_wo_pho(text)
        sequence = self.pho_to_idx(text, add_blank=add_blank)
        return sequence
    
    def text_to_pho(self, text):
        text = english_cleaners_to_phoneme(text)
        return text
    
    def pho_to_idx(self, pho, add_blank=True, interitem = 0):
        sequence = []
        for symbol in pho:
            symbol_id = self.vocab_phones[symbol]
            sequence += [symbol_id]
        
        if add_blank:
            sequence = self._intersperse(sequence, 0)
        
        sequence = torch.LongTensor(sequence)
        return sequence
    
    def text_to_idx(self, text, add_blank=True):
        text = self.text_to_pho(text)
        sequence = self.pho_to_idx(text, add_blank=add_blank)
        return sequence
        
        

        

