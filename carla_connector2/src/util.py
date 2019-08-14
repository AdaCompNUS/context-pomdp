import math

def get_signed_angle_diff(vector1, vector2):
    theta = math.atan2 (vector1.y, vector1.x) - math.atan2 (vector2.y, vector2.x)
    theta = theta * 180.0/3.1415926
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta

