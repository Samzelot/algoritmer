import matplotlib.pyplot as plt

def clamp(value, min_val, max_val):
        return max(min(value, max_val), min_val)

def clamp_arr(i, arr):
     return arr[clamp(i, 0, len(arr) - 1)]

class Spline:
    def __init__(self, controlpoints):
        self.controlpoints = controlpoints
        self.u = list(range(len(self.controlpoints)))
        self.p = 3

    def __call__(self, segment_resolution):
        spline = []
        N = len(self.controlpoints)
        for i in range(N):
            for j in range(segment_resolution):
                u = clamp_arr(i, self.u) + j/segment_resolution*(clamp_arr(i + 1, self.u)- clamp_arr(i, self.u))   
                p =  self.blossom(u, i, self.p)
                spline.append(p)

        return spline

    def blossom(self, u: float, i: int, r: int):

        if r == 0:
            return clamp_arr(i + 1, self.controlpoints)
        
        den = (clamp_arr(i, self.u) - clamp_arr(i + self.p - r + 1, self.u))
        alpha = 0 if den == 0 else (clamp_arr(i, self.u) - u)/den
        
        x1, y1 = self.blossom(u, i - 1, r - 1)
        x2, y2 = self.blossom(u, i, r - 1)
        
        x = x1*(1 - alpha) + x2*alpha
        y = y1*(1 - alpha) + y2*alpha
        return (x, y)
    
    def plot(self):
        x, y = zip(*self.__call__(10))
        plt.plot(x, y, "x-")
        x, y = zip(*self.controlpoints)
        plt.plot(x, y, "*")
        plt.show()

def main():
    spline = Spline([
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (1, 2),
    ])
    spline.plot()

if __name__ == "__main__":
    main()
