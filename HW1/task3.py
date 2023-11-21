
postcards = {
    "Maria": "London",
    "Lorenzo": "Milan",
    "Oleg": "Canberra",
    "Hans": "Calgary",
    "Mark": "Milan",
    "Alex": "Krakow",
    "Julia": "Murmansk"
}

postcards["Petra"] = "Paris"
postcards["Ivan"] = "Moscow"
postcards["Oleg"] = "Sydney"

Cities = set()

for name in postcards:
    city = postcards[name]
    Cities.add(city)
print(postcards)
print('Количество городов:', len(Cities))
print(*Cities, sep = ',')
