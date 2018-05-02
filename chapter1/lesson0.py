import mxnet.ndarray as nd
import mxnet.autograd as ag

x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
with ag.record():
    z = 2 * x * x
z.backward()
print(x.grad)


def f(a):
    b = a * 2
    i = 0
    while nd.norm(b).asscalar() < 1000:
        i += 1
        print(i)
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = b
    else:
        print('100')
        c = 100 * b
    return c


a = nd.random_normal(shape=3)
a.attach_grad()
with ag.record():
    c = f(a)
c.backward()
print(c)
print(a.grad)
