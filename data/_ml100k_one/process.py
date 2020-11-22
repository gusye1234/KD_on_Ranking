
all =[]
with open("test.txt", "r") as f:
    for line in f.readlines():
        data = line.split("\t")
        data = data[:3]
        data = ' '.join(data) + '\n'
        all.append(data)

with open('test.txt','w') as f:
    f.writelines(all)