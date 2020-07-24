import yaml

class Config(dict):
  def __init__(self, config_path):
    super(Config, self).__init__()
    params = yaml.safe_load(open(config_path).read())
    for k, v in params.items():
      self[k] = v
  
  def __getattr__(self, attr):
    return self.get(attr)
