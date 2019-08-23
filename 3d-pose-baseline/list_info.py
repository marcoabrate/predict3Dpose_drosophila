import os

for d in os.listdir('.'):
  if os.path.isdir(d) and "tr_all_te3-24" in d:
    for f in os.listdir(d):
      if f.startswith("info"):
        print(d)
        fp = open(d+"/"+f, 'r')
        print(fp.read()[:257])
        print()
