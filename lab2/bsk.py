import numpy as np
import statistics as stat
import freqency_pl as polish_frequency
import importlib
hash_map = {}
polish_hash_map = {}
change_hash_map = {}
change_hash_map1 = {}
change_hash_map2 = {}
change_hash_map3 = {}
polish_hash_map['A'] = 8.91
polish_hash_map['B'] = 1.47
polish_hash_map['C'] = 3.96
polish_hash_map['D'] = 3.25
polish_hash_map['E'] = 7.66
polish_hash_map['F'] = 1.11
polish_hash_map['G'] = 1.49
polish_hash_map['H'] = 1.11
polish_hash_map['I'] = 8.21
polish_hash_map['J'] = 2.28
polish_hash_map['K'] = 3.51
polish_hash_map['L'] = 2.10
polish_hash_map['M'] = 2.80
polish_hash_map['N'] = 5.52
polish_hash_map['O'] = 6.15
polish_hash_map['P'] = 2.00
polish_hash_map['Q'] = 0.01
polish_hash_map['R'] = 5.47
polish_hash_map['S'] = 5.52
polish_hash_map['T'] = 3.96
polish_hash_map['U'] = 2.50
polish_hash_map['V'] = 0.00
polish_hash_map['W'] = 4.65
polish_hash_map['X'] = 0.00
polish_hash_map['Y'] = 3.76
polish_hash_map['Z'] = 2.28
code_char = ("WXQOWK LXCFRXBCXUAWJO R URXLXCFRXBCXUAWJO J AKAWXQOBM WXNXRUSDEQOWKBCUKBM FDEPACOQK UO NOLDEOWDEROBM")
code_char = code_char.replace(" ", ".")
for ascii_value in range(65,91):
    if ascii_value == 46:
        continue
    else:
        char = chr(ascii_value)
        hash_map[char] = 0
for char in code_char:
    hash_map[char] += 1
for char, count in hash_map.items():
    print(f"{char}: {100*count/len(code_char)}")
print(polish_hash_map)
# Sort the encrypted text frequencies in descending order
sorted_hash = sorted(hash_map.items(), key=lambda x: x[1], reverse=True)

# Sort the Polish language frequencies in descending order
sorted_polish = sorted(polish_hash_map.items(), key=lambda x: x[1], reverse=True)

# Display results side by side
print("\nEncrypted text frequencies vs Polish language frequencies:")
print("Encrypted\t\tPolish")
print("------------------------------")
for i in range(26):
    # Format as percentage with 2 decimal places
    enc_char, enc_freq = sorted_hash[i]
    pol_char, pol_freq = sorted_polish[i]
    
    enc_percent = 100 * enc_freq / len(code_char)
    
    print(f"{enc_char}: {enc_percent:.2f}%\t\t{pol_char}: {pol_freq:.2f}%")

# For frequency matching to help with substitution
print("\nPossible character mappings (by frequency):")
for (enc_char, enc_freq), (pol_char, pol_freq) in zip(sorted_hash, sorted_polish):
    enc_percent = 100 * enc_freq / len(code_char)
    print(f"{enc_char} ({enc_percent:.2f}%) → {pol_char} ({pol_freq:.2f}%)")
     # Use approximate matching instead of exact matching
    if abs(enc_percent - pol_freq) < 0.5:  # tolerance of 0.5%
        change_hash_map[enc_char] = pol_char
        
        # For variant 1, shift one letter forward in the alphabet
        next_char = chr((ord(pol_char) - ord('A') + 1) % 26 + ord('A'))
        change_hash_map1[enc_char] = next_char
        
        # For variant 2, shift one letter backward in the alphabet
        prev_char = chr((ord(pol_char) - ord('A') - 1) % 26 + ord('A'))
        change_hash_map2[enc_char] = prev_char

# Add default mappings for any characters not matched
for char in set(code_char):
    if char not in change_hash_map:
        # Default mapping is the corresponding character from sorted frequencies
        for (enc_char, _), (pol_char, _) in zip(sorted_hash, sorted_polish):
            if enc_char == char:
                change_hash_map[char] = pol_char
                
                # Also create variants
                change_hash_map1[char] = chr((ord(pol_char) - ord('A') + 1) % 26 + ord('A'))
                change_hash_map2[char] = chr((ord(pol_char) - ord('A') - 1) % 26 + ord('A'))
                break
for char in code_char:
    print(change_hash_map[char], end=" ")
print()
for char in code_char:
    print(change_hash_map1[char],end="")
print()
for char in code_char:
    print(change_hash_map2[char],end=" ")

    

