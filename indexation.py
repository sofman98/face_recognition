from face_recognition import face_distance
import numpy as np

a = np.array([1 for i in range(128)])
b = np.array([0 for i in range(128)])
print(a.shape)
distance = face_distance(a, b)
print(distance)