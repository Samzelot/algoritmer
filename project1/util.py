def clamp(value, min_val, max_val):
        return max(min(value, max_val), min_val)

def clamp_arr(i, arr):
     '''Description: 
     Function which returns the left-most node if a node to the left of the grid is called and vice-versa for the right.
      '''
     return arr[clamp(i, 0, len(arr) - 1)]
