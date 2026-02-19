import csv

if __name__ == '__main__':

    header = "v00,v01,v02,v10,v11,v12,v20,v21,v22,normalx,normaly,kappa"

    with open("validation.csv", "r") as f:
        original = f.read()
    with open("data.csv", "w") as f:
        f.write(header + "\n" + original)