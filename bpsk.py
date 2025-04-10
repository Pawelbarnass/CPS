import math
import random

def main():
    n_trials = 1000000  # Liczba prób
    coprime_count = 0
    max_num = 10**9  # Zakres liczb od 1 do max_num

    for _ in range(n_trials):
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        if math.gcd(a, b) == 1:
            coprime_count += 1

    probability = coprime_count / n_trials
    pi_estimate = math.sqrt(6 / probability)
    print(f"Os szacowana wartość π: {pi_estimate}")

if __name__ == "__main__":
    main()