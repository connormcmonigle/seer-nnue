import data_generation
import config as C

def main():
  config = C.Config('config.yaml')
  data_generation.generate_games(config)
  
if __name__ == '__main__':
  main()
