


class randForrest:
  def __init__(self, df, nTrees, nFeat, nSamp, maxDep=None, minLeaf=None):
    self.df = df
    self.X = df.drop(df.columns[-1], axis=1)
    self.Y = df[df.columns[-1]]
    #self.t = nTrees
    self.nFeat = int(np.log2(self.X.shape[1]))
    #self.nFeat = nFeat
    self.size = nSamp
    self.mxDep = maxDep
    self.mnLeaf = minLeaf

    #print(self.nFeat, "sha: ", self.X.shape[1])

    # new data set, random sample with replacement
    self.XY = df.sample(nSamp, replace=True)




    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)

def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)
