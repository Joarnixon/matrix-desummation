from desummation import Desummation

object = Desummation()
A = [[2, 1], [-2, 5]]
C = [[2.07, 0.95], [-2.11, 5.14]]
object.fit(A, 2, n_trials = 2000)
print(object.predict(A))
print(object.W)
print(object.predict(C))
print(object.W)