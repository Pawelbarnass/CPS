#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Sprawdź czy liczba jest potęgą 2
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Znajdź następną potęgę 2
int nextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Rekurencyjna implementacja algorytmu FFT Radix-2
void radix2FFT(const std::vector<std::complex<double>>& x, std::vector<std::complex<double>>& X, int N) {
    const double Pi = 3.14159265358979323846;
    
    if (N == 1) {
        X[0] = x[0];
        return;
    }
    
    int halfN = N / 2;
    std::vector<std::complex<double>> x_even(halfN);
    std::vector<std::complex<double>> x_odd(halfN);
    std::vector<std::complex<double>> X_even(halfN);
    std::vector<std::complex<double>> X_odd(halfN);
    
    // Podział na próbki o parzystych i nieparzystych indeksach
    for (int n = 0; n < halfN; n++) {
        x_even[n] = x[2*n];
        x_odd[n] = x[2*n + 1];
    }
    
    // Rekurencyjne wywołanie FFT dla obu części
    radix2FFT(x_even, X_even, halfN);
    radix2FFT(x_odd, X_odd, halfN);
    
    // Łączenie wyników z wykorzystaniem współczynników obrotowych
    for (int k = 0; k < halfN; k++) {
        std::complex<double> twiddle = std::exp(std::complex<double>(0, -2.0 * Pi * k / N));
        X[k] = X_even[k] + twiddle * X_odd[k];
        X[k + halfN] = X_even[k] - twiddle * X_odd[k];
    }
}

int readfile(const std::string& filename, std::vector<std::complex<double>>& x) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return 0;
    }
    
    double value;
    int count = 0;

    while (file >> value) {
        count++;
    }

    file.clear();
    file.seekg(0, std::ios::beg);
    
    x.resize(count);
    for (int i = 0; i < count && file >> value; i++) {
        x[i] = std::complex<double>(value, 0);
    }
    
    return count;
}

int main() {
    std::cout << "Szukam pliku w katalogu: " << fs::current_path() << std::endl;
    std::vector<std::complex<double>> x;
    int N = readfile("c:\\Users\\pawel\\CPS\\CPS\\random_signal.dat", x);
    
    if (N == 0) {
        std::cerr << "Error: No data read from file." << std::endl;
        return 1;
    }
    std::cout << "Number of points read: " << N << std::endl;
    
    // Sprawdź czy N jest potęgą 2 (wymagane dla Radix-2)
    if (!isPowerOfTwo(N)) {
        int newN = nextPowerOfTwo(N);
        std::cout << "N = " << N << " is not a power of 2. Padding to " << newN << std::endl;
        x.resize(newN, std::complex<double>(0, 0));
        N = newN;
    }
    
    // Przygotowanie wektora wyjściowego
    std::vector<std::complex<double>> X(N);
    
    // Wykonanie FFT algorytmem Radix-2
    radix2FFT(x, X, N);
    
    // Zapisz wyniki do pliku
    std::ofstream outFile("C:\\Users\\pawel\\CPS\\CPS\\lab4\\fft_results.dat");
    if (outFile.is_open()) {
        for (int k = 0; k < N; ++k) {
            X[k] /= sqrt(N); // Dodaj normalizację jeśli jej nie ma
            outFile << k << "\t" << X[k].real() << "\t" << X[k].imag() 
                   << "\t" << std::abs(X[k]) << "\t" << std::arg(X[k]) << std::endl;
                   std::cout << k << "\t" << X[k].real() << "\t" << X[k].imag()
                   << "\t" << std::abs(X[k]) << "\t" << std::arg(X[k]) << std::endl;
        }
        std::cout << "FFT results saved to fft_results.dat" << std::endl;
    }

    return 0;
}