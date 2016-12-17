from matplotlib import pyplot as plt
from sys import argv



def Average(l):
    return sum(l)/len(l)


def CreateAverages(l):
    return zip(*enumerate([Average(x) for x in CreateSlices(l)]))


def CreateSlices(l):
    step_size = 100
    return [l[i:i+step_size] for i in range(0, len(l), step_size)]


path = argv[1]
step = []
losses = []
time = []

with open(path) as f:
    for line in f:
        s, l, t = map(float, line.split())
        step.append(s)
        losses.append(l)
        time.append(t)

step_size = 500

plt.figure(1)
xs, ys = CreateAverages(losses)
plt.scatter(xs, ys, marker='.')
plt.xlim([0, max(xs)])
plt.ylim([min(ys)/1.1, max(ys)*1.1])
plt.title('Average loss per 100 steps')
plt.xlabel('Step/100')
plt.ylabel('Loss')
plt.show()


plt.figure(2)
plt.clf()
xs, ys = CreateAverages(time)
plt.scatter(xs, ys, marker='.')
plt.xlim([0, max(xs)])
plt.ylim([min(ys), Average(ys)])
plt.title('Average time per 100 steps')
plt.xlabel('Step/100')
plt.ylabel('Time')
plt.show()

plt.figure(3)
plt.clf()
xs, ys = (step, losses)
plt.scatter(xs, ys, marker='.')
plt.xlim([0, max(xs)])
plt.ylim([min(ys)/1.1, max(ys)*1.1])
plt.title('Losses per step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
