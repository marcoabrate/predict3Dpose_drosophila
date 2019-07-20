from pickle import load

def read_data(dir):
  with (open(dir, "rb")) as file:
    try:
      return load(file)
    except EOFError:
      return None
