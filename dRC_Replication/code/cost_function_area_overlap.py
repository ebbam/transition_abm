### Plots in python of the BeveridgeCurve using the csv files.
import numpy as np
import pandas as pd
import random
import shapely
from scipy.spatial import ConvexHull
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from sys import argv

#####
# Implementation for the empirical Beveridge Curve
#####np.random.seed(15)
path_local = "../"
path_scripts = "scripts/"
path_exp_sim = "resultfiles/"
path_label = "data/"
path_exp_fig = "reports/fig/"
path_data = "data/"
path_exp_sim = "../results/csv/"
#path where figures are save
path_exp_fig = "../results/fig/"

file_name = "calibration_file.csv"
t_shock = int(argv[1])#100 #argv[1]
cycle_duration = int(argv[2])#130 #argv[2]
tend_transition = t_shock + 14
#argv[4]#25

######
## alpha-shape functions
########
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second

def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst




#######
# Empirical part
#######

#Importing the necessary data and putting it in the right formal
df_u = pd.read_csv(path_local + path_data + "Total_unemployment_BLS1950-2018.csv")
df_v = pd.read_csv(path_local + path_data + "vacancy_rateDec2000.csv")
# Start vacancy rate and unemployment rate since Jan 2001
df_v = df_v.iloc[1:]
df_u = df_u.loc[(df_u['Year']>=2001)]
#putting all values of unemployment in a list
list_unemployment = []
#56 since we excluded previous data points
for i in df_u.index:
    list_unemployment = list_unemployment + list(df_u.loc[i][1:].values)
list_vacancies = list(df_v["JobOpenings"])
# Until Sept 2018
list_unemployment =  list_unemployment[:-3]
list_vacancies = list_vacancies[:-2]
assert(len(list_unemployment) == len(list_vacancies))

# getting the alpha shape
points_empirical = np.zeros([len(list_unemployment),2])
# unemployment in x coordinate and vacancies in y coordinate
for i in range(len(list_unemployment)):
    points_empirical[i, 0] = list_unemployment[i]
    points_empirical[i, 1] = list_vacancies[i]

edges_empirical = alpha_shape(points_empirical, alpha=1, only_outer=True)
bound_edges = stitch_boundaries(edges_empirical)

assert(len(bound_edges) == 1)

x_array = np.zeros(2*len(bound_edges[0]))
y_array = np.zeros(2*len(bound_edges[0]))

points_empirical_alpha = np.zeros([2*len(bound_edges[0]), 2])
count=0
for i, j in bound_edges[0]:
    if count+2 <= len(x_array):
        x_array[count] = points_empirical[[i, j], 0][0]
        y_array[count] = points_empirical[[i, j], 1][0]
        points_empirical_alpha[count][0] = points_empirical[[i, j], 0][0]
        points_empirical_alpha[count][1] = points_empirical[[i, j], 1][0]
        count+=1
        points_empirical_alpha[count][0] = points_empirical[[i, j], 0][1]
        points_empirical_alpha[count][1] = points_empirical[[i, j], 1][1]
        x_array[count] = points_empirical[[i, j], 0][1]
        y_array[count] = points_empirical[[i, j], 1][1]
        count+=1
        #plt.plot(points_empirical[[i, j], 0], points_empirical[[i, j], 1], "r.--")

# area = PolyArea(x_array,y_array)
# plotting
# plt.plot(x_array, y_array, "o--", label="array", alpha=0.8)
# plt.title("area = "+str(area))
# plt.legend()
# plt.show()

#########
# Now get the complex hull for the numerical Beveridge Curve
#########


# save_temp_name = matrix + "_forcalibration_" + "_deltau" + str(δ_u)[3:6] + "v" +str(δ_v)[3:6]# + "tau" + str(int(τ))

df_bc_num = pd.read_csv(path_exp_sim + file_name)

# plt.plot(df_bc_num["unemployment_rate"],df_bc_num["vacancy_rate"])
# plt.show()

u0 = np.array(df_bc_num["unemployment_rate"])[tend_transition:tend_transition + cycle_duration]
v0 = np.array(df_bc_num["vacancy_rate"])[tend_transition:tend_transition + cycle_duration]




#### plot empirical
# plt.plot(u0[:], v0[:], ".")
# plt.show()

# have them in appropriate format
points_numerical = np.zeros([len(u0),2])
# unemployment in x coordinate and vacancies in y coordinate
for i in range(len(u0)):
    points_numerical[i, 0] = u0[i]
    points_numerical[i, 1] = v0[i]

edges_numerical = alpha_shape(points_numerical, alpha=1, only_outer=True)
bound_edges = stitch_boundaries(edges_numerical)

#assert(len(bound_edges) == 1)
#if polygon is simple enough
if len(bound_edges) == 1:
    x_array = np.zeros(2*len(bound_edges[0]))
    y_array = np.zeros(2*len(bound_edges[0]))

    points_numerical_alpha = np.zeros([2*len(bound_edges[0]), 2])
    count=0
    for i, j in bound_edges[0]:
        if count+2 <= len(x_array):
            x_array[count] = points_numerical[[i, j], 0][0]
            y_array[count] = points_numerical[[i, j], 1][0]
            points_numerical_alpha[count][0] = points_numerical[[i, j], 0][0]
            points_numerical_alpha[count][1] = points_numerical[[i, j], 1][0]
            count+=1
            points_numerical_alpha[count][0] = points_numerical[[i, j], 0][1]
            points_numerical_alpha[count][1] = points_numerical[[i, j], 1][1]
            x_array[count] = points_numerical[[i, j], 0][1]
            y_array[count] = points_numerical[[i, j], 1][1]
            count+=1
            #plt.plot(points_empirical[[i, j], 0], points_empirical[[i, j], 1], "r.--")

    # area = PolyArea(x_array,y_array)
    # plotting
    # plt.plot(x_array, y_array, "o--", label="array", alpha=0.8)
    # # plt.title("area = "+str(area))
    # plt.legend()
    # plt.show()

    poly_numerical = Polygon(points_numerical_alpha)
    poly_empirical_alpha = Polygon(points_empirical_alpha)
    # poly_empirical = Polygon(points_empirical)
    # poly_empirical = poly_empirical.buffer(0)
#otherwise buffer it (usuall gives not good cost function anyway)
else:
    poly_numerical = Polygon(points_numerical)
    poly_numerical = poly_numerical.buffer(0)
    poly_empirical_alpha = Polygon(points_empirical_alpha)

try:
    poly_intersect = poly_empirical_alpha.intersection(poly_numerical)
    poly_union = poly_empirical_alpha.union(poly_numerical)
    cost = 1 - poly_intersect.area / poly_union.area
except:
    cost = 1
# poly_intersect = poly_empirical.intersection(poly_numerical)
# poly_union = poly_empirical.union(poly_numerical)
print("cost function", cost, "\n")

f= open(path_exp_sim + "cost.txt","w")
f.write(str(cost))
f.close()
