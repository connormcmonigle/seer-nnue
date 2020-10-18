import math
from PIL import Image
import numpy as np

def weierstrass_sin(x, depth=10):
  return sum(map(lambda i: (2**-i) * math.sin(2**i * x), range(depth)))

 
def weierstrass_cos(x, depth=10):
  return sum(map(lambda i: (2**-i) * math.cos(2**i * x), range(depth)))

def draw(t, data, dim):
  scale = min(dim) / 4
  r = weierstrass_sin(t * weierstrass_cos(t))
  x = int(scale * r * math.cos(t)) + (min(dim)//2)
  y = int(scale * r * math.sin(t)) + (min(dim)//2)
  #print(x, y)
  data[x, y] = tuple(map(lambda p: int(p * 0.3), data[x, y]))


def main():
  dim = (3240, 1080)
  img = Image.new('RGB', dim, 'white')
  pixels = img.load()
  for i in np.linspace(0.0, 100.0, 1000000):
    draw(i, pixels, dim)
  img.show()
  img.save('save.png')

main()