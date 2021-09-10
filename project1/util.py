def clamp(value, min_val, max_val):
        return max(min(value, max_val), min_val)

def clamp_arr(i, arr):
     return arr[clamp(i, 0, len(arr) - 1)]
