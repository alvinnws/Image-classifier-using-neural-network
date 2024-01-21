import csv

findict = {}

with open("cropped-by-semantic-tag/_images_data.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1] not in findict:
            findict[row[1]] = 1
        else:
            findict[row[1]] += 1

for item in findict:
    print(item, findict[item])
