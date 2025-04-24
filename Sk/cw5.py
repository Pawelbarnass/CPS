import csv
import matplotlib.pyplot as plt

# Read data from CSV
x = []
y = []
z = []
with open('Sk\zad5.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        x.append(float(row[0]))  # First column as X-axis
        y.append(float(row[1]))  # Second column as Y-axis
        z.append(float(row[2]))

# Plot the data
plt.figure(figsize=(10, 6))

# Dodanie opisu do pierwszej linii
plt.plot(x, y, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, 
         label='Zadanie 4 - wyniki analizy')  # Dodany parametr label

# Druga linia już ma opis, ale można go doprecyzować
plt.plot(x, z, marker='d', color='r', linestyle='-', linewidth=2, markersize=8, 
         label='Pojedyncza stacja bazowa')  # Ulepszony opis

plt.title('Analiza porównawcza')  # Bardziej opisowy tytuł
plt.xlabel('Częstotliwość [MHz]')  # Poprawiona etykieta osi X
plt.ylabel('zasięg sygnału [km^2]')  # Zakładam, że to jednostka na osi Y
plt.legend()  # Wyświetla legendę z opisami obu linii
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()