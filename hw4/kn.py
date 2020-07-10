# https://visualstudiomagazine.com/articles/2019/01/01/self-organizing-maps-python.aspx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def closet(data, k, map, rows, cols):
    result = (0,0)
    small_dist = 1.0e20
    for i in range(rows):
        for j in range(cols):
            ed = norm_dist(map[i][j], data[k])
            if ed < small_dist:
                small_dist = ed
                result = (i, j)
    return result

def norm_dist(v1, v2):
    return np.linalg.norm(v1 - v2) 

def manhattan_dist(r1, c1, r2, c2):
    return np.abs(r1-r2) + np.abs(c1-c2)

def common(li, n):
    if len(li) == 0: return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(li)):
        counts[li[i]] += 1
    return np.argmax(counts)


    

if __name__=="__main__":
    dim = 13
    rows = 30
    cols = 30
    learn = 0.8
    steps = 5000

    data_file = "./wine_data/wine.data"
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,13),dtype=np.float64)
    data_y = np.loadtxt(data_file, delimiter=",", usecols=[0],dtype=np.int)

    map = np.random.random_sample(size=(rows,cols,dim))
    for s in tqdm(range(steps)):
        pct_left = 1 - ((s * 1) / steps)
        curr_range = (int)(pct_left * rows + cols)
        curr_rate = pct_left * learn

        t = np.random.randint(len(data_x))
        (bmu_row, bmu_col) = closet(data_x, t, map, rows, cols)
        for i in range(rows):
            for j in range(cols):
                if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])

    u_matrix = np.zeros(shape=(rows,cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            v = map[i][j] 
            sum_dists = 0.0; ct = 0
            
            if i-1 >= 0:    
                sum_dists += norm_dist(v, map[i-1][j]); ct += 1
            if i+1 <= rows-1:  
                sum_dists += norm_dist(v, map[i+1][j]); ct += 1
            if j-1 >= 0:  
                sum_dists += norm_dist(v, map[i][j-1]); ct += 1
            if j+1 <= cols-1:   
                sum_dists += norm_dist(v, map[i][j+1]); ct += 1
            
            u_matrix[i][j] = sum_dists / ct

    

    mapping = np.empty(shape=(rows,cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            mapping[i][j] = []

    for t in range(len(data_x)):
        (m_row, m_col) = closet(data_x, t, map, rows, cols)
        mapping[m_row][m_col].append(data_y[t])

    label_map = np.zeros(shape=(rows,cols), dtype=np.int)
    for i in range(rows):
        for j in range(cols):
            label_map[i][j] = common(mapping[i][j], 100)
    plt.imshow(u_matrix, cmap='gray')  
    plt.show()
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 13))
    plt.colorbar()
    plt.show()


