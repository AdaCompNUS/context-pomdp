import numpy as np
# import scipy
# import scipy.interpolate as interp
import math


# def interpolate_polyline(polyline, num_points):
#     duplicates = []
#     for i in range(1, len(polyline)):
#         if np.allclose(polyline[i], polyline[i - 1]):
#             duplicates.append(i)
#     if duplicates:
#         polyline = np.delete(polyline, duplicates, axis=0)
#     tck, u = interp.splprep(polyline.T, s=0)
#     u = np.linspace(0.0, 1.0, num_points)
#     return np.column_stack(interp.splev(u, tck))


def distance(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def move_along(A, B, t, step=0.03):
    dist = distance(A, B)
    if t + step <= dist:
        t = t + step
        r = t / dist
        point = [A[0] * (1 - r) + B[0] * r, A[1] * (1 - r) + B[1] * r]
        return True, point, t
    else:
        return False, None, t + step - dist  # t for next segment


def smoothing(points, step=0.3):
    t = 0
    i = 0
    path = []
    last_point = None
    while i < len(points):
        point = points[i]
        if last_point is not None:
            suc, mid, new_t = move_along(last_point, point, t)
            if suc:
                t = new_t
                path.append(mid)
            else:  # at the end of the current segment
                last_point = point
                i += 1  # move to next segment
                t = new_t - step
        else:
            # at the start of path
            path.append(point)
            last_point = point
            i += 1

    return np.array(path)


import matplotlib.pyplot as plt


def plot(A, B, C):
    plt.figure()
    plt.plot(A[:, 0], A[:, 1], 'bo')
    # plt.plot(B[:,0], B[:,1], 'g')
    plt.plot(C[:, 0], C[:, 1], 'r')
    plt.title('Interpolation results')
    plt.show()


if __name__ == '__main__':
    pass
    # way_points = [[0.0, 1.0], [2.0, 3.3], [5.0, 3.0], [5.0, 6.0], [8.0,
    #                                                                9.0], [6.0, 9.5], [6.8, 10.5]]
    # for point in way_points:
    #     print(point)
    #
    # print('=====================================')
    # way_points = np.array(way_points)
    # smooth_points = interpolate_polyline(way_points, 50)
    # last_point = None
    # for point in smooth_points:
    #     print(point)
    #     if last_point:
    #         dist = distance(last_point, point)
    #         print('dist {}'.format(dist))
    #     last_point = [point[0], point[1]]
    #
    # print('=====================================')
    # even_points = smoothing(smooth_points)
    #
    # last_point = None
    # for point in even_points:
    #     print(point)
    #     if last_point:
    #         dist = distance(last_point, point)
    #         print('dist {}'.format(dist))
    #     last_point = [point[0], point[1]]
    #
    # plot(way_points, smooth_points, even_points)