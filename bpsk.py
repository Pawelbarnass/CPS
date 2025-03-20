import numpy as np
import statistics as stat
import freq_pl
import importlib
hash_map = {}
polish_hash_map = freq_pl.polish_frequency()
code_char = ("WXQOWK LXCFRXBCXUAWJO R URXLXCFRXBCXUAWJO J AKAWXQOBM WXNXRUSDEQOWKBCUKBM FDEPACOQK UO NOLDEOWDEROBM")
code_char = code_char.replace(" ", "")
for ascii_value in range(65,91):
    char = chr(ascii_value)
    hash_map[char] = 0
for char in code_char:
    hash_map[char] += 1
for char, count in hash_map.items():
    print(f"{char}: {100*count/len(code_char)}")