


class datalogger:
  def __init__(self, enable):
    self.on = enable # enable flag
    self.W1 = list()
    self.B1 = list()
    self.Z1 = list()
    self.Y1 = list()
    self.W2 = list()
    self.B2 = list()
    self.Z2 = list()
    self.Yest = list()
    self.iters = list()
    self.W1update = list()
    self.B1update = list()
    self.W2update = list()
    self.B2update = list()
    self.accHist = list()
    return
