import matplotlib.pyplot as plt

xs, ys, zs = [], [], []
colors = []

with open("reconstruction_sparse.ply", "r") as f:
    header = True
    for line in f:
        if header:
            if line.strip() == "end_header":
                header = False
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        x, y, z = map(float, parts[:3])
        r, g, b = map(int, parts[3:6])

        xs.append(x)
        ys.append(y)
        zs.append(z)

        colors.append((r/255, g/255, b/255))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(xs, ys, zs, c=colors, s=25)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])

plt.show()
