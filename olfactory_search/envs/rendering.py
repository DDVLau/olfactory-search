import numpy as np
import math


CELL_TO_IDX = {
    None: 0,
    "odor": 1,
    "start": 2,
    "source": 3,
    "agent": 4,
}

ROUTE_TO_IDX = {
    None: 0,
    "12": 1,
    "3": 2,
    "6": 3,
    "9": 4,
}


COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "black": np.array([0, 0, 0]),
    "grey": np.array([100, 100, 100]),
    "hit3": np.array([158, 0, 66]),
    "hit2": np.array([252, 226, 85]),
    "hit1": np.array([50, 187, 197]),
}

ROUTE = {
    "12": (0.5, 0.0, 0.5, 0.5, 0.05),
    "6": (0.5, 1.0, 0.5, 0.5, 0.05),
    "9": (0.0, 0.5, 0.5, 0.5, 0.05),
    "3": (1.0, 0.5, 0.5, 0.5, 0.05),
}


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def adjust_alpha(color, a):
    return color * a + (1.0 - a) * 255


class OdorCell:
    def __init__(self, hit:int|float=-2, type:str="odor"):
        self.hit = hit
        self.actions = {"12": False, "3": False, "6": False, "9": False}
        # self.last_action = None if last_action is None else compare_vectors(last_action)
        # self.next_action = None if next_action is None else compare_vectors(next_action)
        self.type = type

    def dict_to_int(self):
        return int(
            "".join(
                [str(int(self.actions[action])) for action in ["12", "3", "6", "9"]]
            ),
            2,
        )

    def encode(self) -> tuple[int, int, int, int]:
        """Encode the a description of this object"""
        return (CELL_TO_IDX[self.type], self.dict_to_int(), self.hit)

    def add_action(self, v1, v2):
        action = None
        if v1 == 0 and v2 == 1:
            action = "12"
        elif v1 == 0 and v2 == -1:
            action = "6"
        elif v1 == 1 and v2 == 0:
            action = "9"
        elif v1 == -1 and v2 == 0:
            action = "3"

        if action is not None:
            self.actions[action] = True

    def render(self, img):
        # Blank cell
        if self.hit == -2:
            return img

        # Draw the lower layer
        if self.hit == -1:
            color = COLORS["red"]
        elif self.hit == 0:
            color = COLORS["grey"]
        elif self.hit == 1:
            color = COLORS["hit1"]
        elif self.hit == 2:
            color = COLORS["hit2"]
        elif self.hit == 3:
            color = COLORS["hit3"]
        else:
            color = adjust_alpha(COLORS["hit1"], np.clip(self.hit, 0, 1)) # Scale for better illustration

        fill_coords(img, point_in_circle(0.5, 0.5, 0.4), color=color)

        # Draw the upper layer
        for keys in self.actions.keys():
            if self.actions[keys]:
                fill_coords(img, point_in_line(*ROUTE[keys]), color=COLORS["black"])

        if self.type == "start":
            fill_coords(
                img, point_in_line(0.0, 0.0, 1.0, 1.0, 0.02), color=COLORS["black"]
            )
            fill_coords(
                img, point_in_line(1.0, 0.0, 0.0, 1.0, 0.02), color=COLORS["black"]
            )
        elif self.type == "source":
            fill_coords(
                img,
                point_in_triangle((0.35, 0.2), (0.35, 0.6), (0.75, 0.4)),
                color=COLORS["red"],
            )
            fill_coords(
                img, point_in_line(0.35, 0.2, 0.35, 0.8, 0.02), color=COLORS["red"]
            )
        
        if self.type == "agent":
            fill_coords(img, point_in_circle(0.5, 0.5, 0.5), color=COLORS["grey"])
            fill_coords(img, point_in_circle(0.5, 0.5, 0.4), color=COLORS["black"])

        return img


# class StartCell(OdorCell):
#     def __init__(self, hit=0, last_action=None, next_action=None) -> None:
#         super().__init__(hit=hit, last_action=None, next_action=next_action)
#         self.type = "start"

#     def render(self, img):
#         img = super().render(img)
#         fill_coords(img, point_in_line(0.0, 0.0, 1.0, 1.0, 0.02), color=COLORS["black"])
#         fill_coords(img, point_in_line(1.0, 0.0, 0.0, 1.0, 0.02), color=COLORS["black"])
#         return img


# class SourceCell(OdorCell):
#     def __init__(self, hit=0, last_action=None, next_action=None) -> None:
#         super().__init__(hit=hit, last_action=last_action, next_action=None)
#         self.type = "source"

#     def render(self, img):
#         fill_coords(img, point_in_line(0.0, 0.5, 1.0, 0.5, 0.03), color=COLORS["black"])
#         fill_coords(img, point_in_line(0.5, 0.2, 0.5, 0.8, 0.02), color=COLORS["black"])
#         return img


# class AgentCell(OdorCell):
#     def __init__(self, hit=0, last_action=None, next_action=None) -> None:
#         super().__init__(hit=hit, last_action=last_action, next_action=None)
#         self.type = "agent"

#     def render(self, img):
#         fill_coords(img, point_in_circle(0.5, 0.5, 0.5), color=COLORS["grey"])
#         fill_coords(img, point_in_circle(0.5, 0.5, 0.4), color=COLORS["black"])
#         return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = 255 * np.ones((64, 64, 3), dtype=np.uint8)
    # for i in range(4):
    #     cell = OdorCell(hit=i)
    #     img = cell.render(img)
    #     plt.imshow(img)
    #     plt.show()

    cell = OdorCell(hit=0, type="source")
    cell.add_action(0, 1)
    cell.add_action(1, 0)
    print(cell.encode())
    img = cell.render(img)
    plt.imshow(img)
    plt.show()
