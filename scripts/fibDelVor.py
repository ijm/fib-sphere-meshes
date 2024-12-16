import numpy as np
import numpy.typing as npt


# Constans and setup.
def m(x: int) -> float:
    return (x + np.sqrt(x * x + 4.)) / 2.


Phi: float = m(1)
pi2: float = np.pi * 2.0
root5: float = np.sqrt(5)

index_t = int
coord2D_t = tuple[float, float]
coord3D_t = tuple[float, float, float]
octogon_t = tuple[int, int, int, int, int, int, int, int]
matrix2x2 = npt.NDArray

polePoly: list[list[int]] = [
    [1, 2, 5, 8, 3],
    [0, 3, 6, 9, 4, 2],
    [0, 1, 4, 7, 10, 5],
    [0, 1, 6, 11, 16, 8],
    [1, 2, 7, 12, 17, 9],
]


# Primative helper functions.
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def fract(x: float) -> float:
    return x % 1.0


def hat(v):
    return v / np.linalg.norm(v)


def cmpswap(arr: npt.NDArray, center: int, n: int, a: int, b: int) -> None:
    pa = (arr[a] - center + Phi / 2) % Phi
    pb = (arr[b] - center + Phi / 2) % Phi

    if pa > pb:
        arr[b], arr[a] = (arr[a], arr[b])

# ----
# Index Mapings


def uvOfIndex(index: index_t, n: int) -> coord2D_t:
    return fract((index) / Phi), (index + 0.5) / n


def phiZOfuv(uv: coord2D_t) -> coord2D_t:
    u, v = uv
    return (pi2 * u, 1 - 2 * v)


def uvOfPhiZ(phiZ: coord2D_t) -> coord2D_t:
    phi, z = phiZ
    return (phi / pi2, 0.5 * (1 - z))


def sphereOfPhiZ(phiZ: coord2D_t) -> coord3D_t:
    phi, z = phiZ
    sin_theta = np.sqrt(1 - z * z)

    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta

    return (x, y, z)


def phiZOfIndex(index: index_t, n: float) -> coord2D_t:
    phi = 2 * np.pi * fract(index * (Phi - 1))
    z = 1. - (2. * index + 1.) / n  # * np.reciprocal(float(n))

    return (phi, z)


def kOfZ(z: float, n: int) -> int:
    return max(2, np.floor(
        np.log(n * np.pi * root5 * (1. - z * z)) / np.log(Phi * Phi)
    ))


def delta(index1: index_t, index2: index_t, n: int) -> float:
    dphi = 2 * np.pi * fract((index2 - index1) * (Phi - 1))
    z1 = 1. - (2. * index1 + 1.) / n
    z2 = 1. - (2. * index2 + 1.) / n

    cp2 = np.cos(dphi)
    nz1 = 1 - z1 * z1
    nz2 = 1 - z2 * z2
    sz12 = np.sqrt(nz1 * nz2)

    return np.sqrt(2 * (1 - sz12 * cp2 - z1 * z2))


def basisOfK(k: int, n: int) -> matrix2x2:
    Fk = np.power(Phi, k) / root5
    F0 = np.round(Fk)
    F1 = np.round(Fk * Phi)

    B = np.array([[
        2 * np.pi * fract((F0 + 1) * (Phi - 1)) - 2 * np.pi * (Phi - 1.),
        2 * np.pi * fract((F1 + 1) * (Phi - 1)) - 2 * np.pi * (Phi - 1.),
    ], [
        -2 * F0 / n,
        -2 * F1 / n
    ]])

    return B


def basisOfZ(z: float, n: int) -> matrix2x2:
    return basisOfK(kOfZ(z, n), n)


def indexOfZ(cosTheta: float, n: int) -> index_t:
    cosTheta = np.clip(cosTheta, -1, 1) * 2 - cosTheta
    i = np.floor(n * 0.5 - cosTheta * n * 0.5)
    return i


# Top level Forward and Inverse Functions.
def SF(index: index_t, n: int) -> coord3D_t:
    return sphereOfPhiZ(phiZOfuv(uvOfIndex(index, n)))


def inverseSF(p: coord3D_t, n: int) -> index_t:
    x, y, z = p

    phi = np.arctan2(y, x)

    B = basisOfZ(z, n)
    cc = np.array([phi, z - (1. - 1. / n)])

    rr = np.linalg.solve(B, cc)

    c = np.floor(rr)
    d = np.infty
    j = 0

    for s in range(4):
        cosTheta1 = np.dot(B[1], np.array([s % 2, s // 2]) + c) + (1. - 1. / n)
        i = indexOfZ(cosTheta1, n)

        phi = 2 * np.pi * fract(i * (Phi - 1))
        cosTheta2 = 1 - (2 * i + 1) / n

        sinTheta = np.sqrt(1 - cosTheta2 * cosTheta2)

        q = np.array([
            np.cos(phi) * sinTheta,
            np.sin(phi) * sinTheta,
            cosTheta2
        ])

        squaredDistance = np.dot(q - p, q - p)
        if (squaredDistance < d):
            d = squaredDistance
            j = i

    return j

# ----
# Mesh functionsA


# Mesh Helper functions.
def xyzOfIndexes(pv: list[index_t], n: int) -> list[coord3D_t]:
    """Compute the 3D vertex points from indexes."""
    return [SF(v, n) for v in pv]


def dMaxOfN(n: int) -> float:
    """Calculate max mesh edge length"""
    SFn1 = np.array(SF(1, n))
    SFn2 = np.array(SF(2, n))
    SFn4 = np.array(SF(4, n))

    # v = np.cross(SF(2, n) - SF(1, n), SF(4, n) - SF(1, n))
    v = np.cross(SFn2 - SFn1, SFn4 - SFn1)
    v /= np.linalg.norm(v)

    return np.arccos(np.dot(v, SFn1))


def circumcenter(A: coord3D_t, B: coord3D_t, C: coord3D_t) -> tuple[coord3D_t, float]:
    """Compute the circumcenter of 3 x 3D points."""
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)

    # N = np.cross(AB, AC)
    dAB = np.dot(AB, AB)
    dAC = np.dot(AC, AC)
    dABAC = np.dot(AB, AC)
    # dN = np.dot(N, N)

    D = np.array([[dAB, dABAC], [dABAC, dAC]])
    x = np.array([0.5 * dAB, 0.5 * dAC])

    (a, b) = np.linalg.solve(D, x)  # same asinv(D) * x

    R = np.array(A) + (a * AB + b * AC)

    return (tuple(R), 0)


# Main workhorse functions.

def candidateOctFromIndex(index: index_t, n: int, dmax: float) -> octogon_t:
    """
    Find the upto eight 'nearest' points around the index point, using the
    property that they must be a Fibonacci step from each other.
    Sort them into a polygon using two properties: 1. four must have z above
    and four must have z below the center index, and 2. each of the four are
    monatonic in their phi value.
    Note: the poles are hard, so don't use this at the poles.
    """

    f = Phi

    for _ in range(100):  # FIX - untidy and Might break for spheres over 100,000,000 points.
        f *= Phi
        Fn = int(np.round(f / root5))
        pi = index + Fn
        pn = index - Fn

        if pi >= n or delta(pi, index, n) < dmax * 2:
            break
        if pn < 0 or delta(pn, index, n) < dmax * 2:
            break

    P = np.array([1., Phi, Phi * Phi, Phi * Phi * Phi]) / root5
    F = np.round(P * f).astype(int)

    pi4 = index + F
    pn4 = index - F

    pi4[pi4 >= n] = pi
    pn4[pn4 < 0] = pn

    cmpswap(pi4, index, n, 0, 2)
    cmpswap(pi4, index, n, 1, 3)
    cmpswap(pi4, index, n, 0, 1)
    cmpswap(pi4, index, n, 2, 3)
    cmpswap(pi4, index, n, 1, 2)

    cmpswap(pn4, index, n, 2, 0)
    cmpswap(pn4, index, n, 3, 1)
    cmpswap(pn4, index, n, 1, 0)
    cmpswap(pn4, index, n, 3, 2)
    cmpswap(pn4, index, n, 2, 1)

    return tuple(pi4) + tuple(pn4)


def polyOfOct(p: octogon_t, index: index_t, n: int) -> list[int]:
    """
    Given upto eight points in a tri-mesh poly around an index point, filter
    out invalid Delaunay triangles by checking the sum of oposite angles.
    """
    # tris are OAB OBC
    # alpha = dot(AO, AB)
    # gamma = dot(CO, CB)
    pv = [p[j] for j in range(8) if p[j] != p[(j + 1) & 7]]
    sxyz = [np.array(SF(v, n)) for v in pv]
    N = len(pv)

    so = np.array(SF(index, n))

    alphas = [np.arccos(np.dot(hat(so - sxyz[j]), hat(sxyz[(j + 1) % N] - sxyz[j]))) for j in range(N)]
    gammas = [np.arccos(np.dot(hat(so - sxyz[j]), hat(sxyz[(j - 1) % N] - sxyz[j]))) for j in range(N)]

    pv = [pv[j] for j in range(N) if alphas[(j - 1) % N] + gammas[(j + 1) % N] <= np.pi]

    return pv


def vorOfDelTriXYZ(xyz_centre: coord3D_t, xyz: list[coord3D_t]) -> list[coord3D_t]:
    """Compute the 3D Voronoi points as the circumcenters of the local Delaunay mesh 3D verticies."""

    N = len(xyz)
    return [circumcenter(xyz_centre, xyz[i], xyz[(i + 1) % N])[0] for i in range(N)]


def polyFromIndex(index: index_t, n: int) -> list[int]:
    """Compute the local Delaunay mesh around an index point as indexes."""
    if index < 5:
        poly = polePoly[index]
    else:
        dmax = dMaxOfN(n)
        candidates = candidateOctFromIndex(index, n, dmax)
        poly = polyOfOct(candidates, index, n)

    return poly


# ----
# Testing...

def main():
    N = 64
    i = 29

    poly = polyFromIndex(i, N)
    print(f"{poly=}")

    cnt = SF(i, N)
    xyzs = vorOfDelTriXYZ(cnt, xyzOfIndexes(poly, N))
    print(f"{xyzs=}")


if __name__ == "__main__":
    main()
