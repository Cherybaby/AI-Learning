import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve


@dataclass
class HardPoints:
    A1: np.ndarray
    B1: np.ndarray
    A2: np.ndarray
    B2: np.ndarray
    A3: np.ndarray
    B3: np.ndarray
    A4: np.ndarray
    B4: np.ndarray
    A5: np.ndarray
    B5: np.ndarray
    C: np.ndarray
    C1: np.ndarray
    LUP: np.ndarray
    LLWR: np.ndarray
    SCLP: np.ndarray


def _rotation_matrix_for_constraint(a: np.ndarray) -> np.ndarray:
    """Rotation matrix used by XP_algorithm.m (note d23 sign keeps MATLAB behavior)."""
    d11 = math.cos(a[0]) * math.cos(a[1])
    d12 = -math.sin(a[0]) * math.cos(a[1])
    d13 = math.sin(a[1])
    d21 = math.sin(a[0]) * math.cos(a[2]) + math.cos(a[0]) * math.sin(a[1]) * math.sin(a[2])
    d22 = math.cos(a[0]) * math.cos(a[2]) - math.sin(a[0]) * math.sin(a[1]) * math.sin(a[2])
    d23 = -math.cos(a[1]) * math.sin(a[2])
    d31 = math.sin(a[0]) * math.sin(a[2]) - math.cos(a[0]) * math.sin(a[1]) * math.cos(a[2])
    d32 = math.cos(a[0]) * math.sin(a[2]) + math.sin(a[0]) * math.sin(a[1]) * math.cos(a[2])
    d33 = math.cos(a[1]) * math.cos(a[2])
    return np.array(
        [[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]], dtype=float
    )


def _rotation_matrix_for_main(x: np.ndarray) -> np.ndarray:
    """Rotation matrix used in RR_FIVE_LINK_SUSP.m main loop."""
    d11 = math.cos(x[0]) * math.cos(x[1])
    d12 = -math.sin(x[0]) * math.cos(x[1])
    d13 = math.sin(x[1])
    d21 = math.sin(x[0]) * math.cos(x[2]) + math.cos(x[0]) * math.sin(x[1]) * math.sin(x[2])
    d22 = math.cos(x[0]) * math.cos(x[2]) - math.sin(x[0]) * math.sin(x[1]) * math.sin(x[2])
    d23 = math.cos(x[1]) * math.sin(x[2])
    d31 = math.sin(x[0]) * math.sin(x[2]) - math.cos(x[0]) * math.sin(x[1]) * math.cos(x[2])
    d32 = math.cos(x[0]) * math.sin(x[2]) + math.sin(x[0]) * math.sin(x[1]) * math.cos(x[2])
    d33 = math.cos(x[1]) * math.cos(x[2])
    return np.array(
        [[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]], dtype=float
    )


def _transform_point(E: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return E @ p + Y


def _xp_algorithm(a: np.ndarray, Ctz: float, hp: HardPoints) -> np.ndarray:
    """Python equivalent of XP_algorithm.m for fsolve residuals."""
    E1 = _rotation_matrix_for_constraint(a)
    Y1 = np.array(
        [
            a[3] - np.dot(E1[0, :], hp.C),
            a[4] - np.dot(E1[1, :], hp.C),
            Ctz - np.dot(E1[2, :], hp.C),
        ],
        dtype=float,
    )

    B11 = _transform_point(E1, Y1, hp.B1)
    B22 = _transform_point(E1, Y1, hp.B2)
    B33 = _transform_point(E1, Y1, hp.B3)
    B44 = _transform_point(E1, Y1, hp.B4)
    B55 = _transform_point(E1, Y1, hp.B5)

    ss1 = np.linalg.norm(B11 - hp.A1) - np.linalg.norm(hp.B1 - hp.A1)
    ss2 = np.linalg.norm(B22 - hp.A2) - np.linalg.norm(hp.B2 - hp.A2)
    ss3 = np.linalg.norm(B33 - hp.A3) - np.linalg.norm(hp.B3 - hp.A3)
    ss4 = np.linalg.norm(B44 - hp.A4) - np.linalg.norm(hp.B4 - hp.A4)
    ss5 = np.linalg.norm(B55 - hp.A5) - np.linalg.norm(hp.B5 - hp.A5)
    return np.array([ss1, ss2, ss3, ss4, ss5], dtype=float)


def _solve_y_with_fixed_length(Bt: np.ndarray, A: np.ndarray, L: float) -> float:
    """Equivalent to symbolic solve for y and MATLAB min(root1, root2) branch choice."""
    delta = L * L - (Bt[0] - A[0]) ** 2 - (Bt[2] - A[2]) ** 2
    if delta < 0 and abs(delta) < 1e-8:
        delta = 0.0
    if delta < 0:
        raise ValueError(f"Invalid geometry, negative square term: {delta}")
    root = math.sqrt(delta)
    y1 = A[1] - root
    y2 = A[1] + root
    return min(y1, y2)


def _solve_contact_point(Ct: np.ndarray, C1t: np.ndarray, tire_radius: float, Ctz: float) -> np.ndarray:
    """Equivalent contact-point construction in MATLAB section '接地点坐标'."""
    Jz = Ctz - 10.0
    wheel = C1t - Ct

    dz = Jz - Ct[2]
    rhs = -wheel[2] * dz

    # Solve line-circle intersection in (dx, dy):
    # wheel_x * dx + wheel_y * dy = rhs
    # dx^2 + dy^2 = tire_radius^2 - dz^2
    r2_xy = tire_radius * tire_radius - dz * dz
    if r2_xy < 0 and abs(r2_xy) < 1e-8:
        r2_xy = 0.0
    if r2_xy < 0:
        raise ValueError(f"No real contact solution, negative planar radius squared: {r2_xy}")

    wx, wy = wheel[0], wheel[1]
    norm_w = math.hypot(wx, wy)
    if norm_w < 1e-12:
        raise ValueError("Degenerate wheel axis projection in XY plane")

    # Closest point on the line to origin in (dx,dy).
    d0x = (rhs / (norm_w * norm_w)) * wx
    d0y = (rhs / (norm_w * norm_w)) * wy

    # Perpendicular direction in line space.
    px = -wy / norm_w
    py = wx / norm_w

    off2 = r2_xy - (d0x * d0x + d0y * d0y)
    if off2 < 0 and abs(off2) < 1e-8:
        off2 = 0.0
    if off2 < 0:
        raise ValueError(f"No real line-circle intersection, negative offset squared: {off2}")
    off = math.sqrt(off2)

    dx1, dy1 = d0x + off * px, d0y + off * py
    dx2, dy2 = d0x - off * px, d0y - off * py

    J1 = np.array([Ct[0] + dx1, Ct[1] + dy1, Jz], dtype=float)
    J2 = np.array([Ct[0] + dx2, Ct[1] + dy2, Jz], dtype=float)

    # MATLAB uses TT_CTR as average of two symbolic solutions.
    TT_CTR = 0.5 * (J1 + J2)

    # Solve norm(Ct + t*(TT_CTR - Ct) - Ct) = tire_radius.
    v = TT_CTR - Ct
    vv = float(np.dot(v, v))
    if vv < 1e-20:
        raise ValueError("Degenerate TT_CTR direction vector")

    t_abs = tire_radius / math.sqrt(vv)
    # Keep the positive branch to match typical MATLAB solve scalar behavior.
    t = t_abs

    CONT = Ct + t * v
    return CONT


def _solve_rch_point(yc: float, zc: float, ya: float, za: float, yb: float, zb: float) -> tuple[float, float]:
    """Equivalent to solve(sd1, sd2, y0, z0) with linearized equal-distance equations."""
    a11 = 2.0 * (yc - ya)
    a12 = 2.0 * (zc - za)
    b1 = yc * yc + zc * zc - ya * ya - za * za

    a21 = 2.0 * (ya - yb)
    a22 = 2.0 * (za - zb)
    b2 = ya * ya + za * za - yb * yb - zb * zb

    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        raise ValueError("Degenerate points for RCH solving (near-collinear in y-z projection)")

    y0 = (b1 * a22 - a12 * b2) / det
    z0 = (a11 * b2 - b1 * a21) / det
    return y0, z0


def run_simulation() -> dict[str, np.ndarray]:
    # MATLAB script constants.
    BOUstroke = 100.0
    REBstroke = -100.0
    tire_radius = 352.0
    step = 5.0

    hp = HardPoints(
        A1=np.array([4313.0, -428.0, 1054.5], dtype=float),
        B1=np.array([4365.008, -715.386, 1065.326], dtype=float),
        A2=np.array([4588.0, -357.0, 1090.5], dtype=float),
        B2=np.array([4444.123, -697.692, 1111.512], dtype=float),
        A3=np.array([4242.0, -413.0, 863.5], dtype=float),
        B3=np.array([4391.608, -730.852, 801.842], dtype=float),
        A4=np.array([4709.907, -292.376, 865.386], dtype=float),
        B4=np.array([4535.801, -714.774, 889.46], dtype=float),
        A5=np.array([4316.0, -446.0, 936.5], dtype=float),
        B5=np.array([4292.353, -722.178, 943.276], dtype=float),
        C=np.array([4421.553, -825.1, 972.284], dtype=float),
        C1=np.array([4421.811, -745.143, 970.287], dtype=float),
        LUP=np.array([4630.495, -640.275, 1044.481], dtype=float),
        LLWR=np.array([4617.547, -667.336, 882.149], dtype=float),
        SCLP=np.array([4881.0, -480.0, 1002.5], dtype=float),
    )

    UPRBOU = BOUstroke + hp.C[2]
    LWRREB = REBstroke + hp.C[2]

    # MATLAB style inclusive scan.
    ctz_values = np.arange(LWRREB, UPRBOU + step * 0.5, step, dtype=float)

    Ctz_plot = []
    Ct_write = []
    KNU_write = []
    Blt_write = []
    B2t_write = []
    B3t_write = []
    B4t_write = []
    B5t_write = []
    camber = []
    toe = []
    CONT_PONT = []

    x_init = np.array([0.0, 0.0, 0.0, 4421.553, -825.1], dtype=float)
    last_opt: dict[str, float | int | np.ndarray] | None = None

    for Ctz in ctz_values:
        # Match MATLAB behavior: each Ctz step uses the same initial guess.
        x, info, ier, msg = fsolve(
            func=lambda a: _xp_algorithm(a, Ctz, hp),
            x0=x_init,
            xtol=1e-12,
            maxfev=10000,
            full_output=True,
        )
        # SciPy may return ier=3 for "xtol too small" while residuals are already tiny.
        # Treat this as converged when the residual norm is sufficiently small.
        residual_norm = float(np.linalg.norm(info["fvec"]))
        if ier not in (1, 3) or residual_norm > 1e-8:
            raise RuntimeError(
                f"fsolve failed at Ctz={Ctz:.6f}: ier={ier}, residual={residual_norm:.3e}, msg={msg}"
            )

        last_opt = {
            "Ctz": float(Ctz),
            "x": x.copy(),
            "ier": int(ier),
            "residual_norm": residual_norm,
            "fvec": info["fvec"].copy(),
        }

        E = _rotation_matrix_for_main(x)
        Y = np.array(
            [
                x[3] - np.dot(E[0, :], hp.C),
                x[4] - np.dot(E[1, :], hp.C),
                Ctz - np.dot(E[2, :], hp.C),
            ],
            dtype=float,
        )

        B11t = _transform_point(E, Y, hp.B1)
        B22t = _transform_point(E, Y, hp.B2)
        B33t = _transform_point(E, Y, hp.B3)
        B44t = _transform_point(E, Y, hp.B4)
        B55t = _transform_point(E, Y, hp.B5)
        C11t = _transform_point(E, Y, hp.C1)

        # Match MATLAB branch rule: solve y and keep min(root1, root2).
        B11t[1] = _solve_y_with_fixed_length(B11t, hp.A1, np.linalg.norm(hp.B1 - hp.A1))
        B22t[1] = _solve_y_with_fixed_length(B22t, hp.A2, np.linalg.norm(hp.B2 - hp.A2))
        B33t[1] = _solve_y_with_fixed_length(B33t, hp.A3, np.linalg.norm(hp.B3 - hp.A3))
        B44t[1] = _solve_y_with_fixed_length(B44t, hp.A4, np.linalg.norm(hp.B4 - hp.A4))
        B55t[1] = _solve_y_with_fixed_length(B55t, hp.A5, np.linalg.norm(hp.B5 - hp.A5))

        Ct = np.array([x[3], x[4], Ctz], dtype=float)
        C1t = C11t.copy()

        Ctz_plot.append(Ctz - hp.C[2])
        Ct_write.append(Ct)
        KNU_write.append(C1t)
        Blt_write.append(B11t)
        B2t_write.append(B22t)
        B3t_write.append(B33t)
        B4t_write.append(B44t)
        B5t_write.append(B55t)

        delta_z = Ctz - C1t[2]
        delta_y = Ct[1] - C1t[1]
        delta_x = Ct[0] - C1t[0]
        camber.append(math.degrees(math.atan(delta_z / delta_y)))
        toe.append(math.degrees(math.atan(delta_x / delta_y)))

        cont = _solve_contact_point(Ct, C1t, tire_radius, Ctz)
        CONT_PONT.append(cont)

    CONT_PONT_arr = np.array(CONT_PONT, dtype=float)

    # Side view roll center height calculation (same point-pairing rule as MATLAB).
    num = int((UPRBOU - LWRREB) / step + 1)
    aa = int(40 / step)

    ab = []
    RCH = []
    ycc, zcc, yaa, zaa, ybb, zbb = [], [], [], [], [], []
    WC_RCH = []

    for i in range(0, num - 2 * aa):
        ab.append(i + 1)  # MATLAB index starts from 1

        yc = CONT_PONT_arr[i, 1]
        zc = CONT_PONT_arr[i, 2]
        ya = CONT_PONT_arr[i + aa, 1]
        za = CONT_PONT_arr[i + aa, 2]
        yb = CONT_PONT_arr[i + 2 * aa, 1]
        zb = CONT_PONT_arr[i + 2 * aa, 2]

        ycc.append(yc)
        zcc.append(zc)
        yaa.append(ya)
        zaa.append(za)
        ybb.append(yb)
        zbb.append(zb)

        y00, z00 = _solve_rch_point(yc, zc, ya, za, yb, zb)
        h = ya * (z00 - za) / (ya - y00)
        RCH.append(h)
        WC_RCH.append(Ctz_plot[i + aa])

    return {
        "Ctz_plot": np.array(Ctz_plot, dtype=float),
        "Ct_write": np.array(Ct_write, dtype=float),
        "KNU_write": np.array(KNU_write, dtype=float),
        "Blt_write": np.array(Blt_write, dtype=float),
        "B2t_write": np.array(B2t_write, dtype=float),
        "B3t_write": np.array(B3t_write, dtype=float),
        "B4t_write": np.array(B4t_write, dtype=float),
        "B5t_write": np.array(B5t_write, dtype=float),
        "camber": np.array(camber, dtype=float),
        "toe": np.array(toe, dtype=float),
        "CONT_PONT": CONT_PONT_arr,
        "ab": np.array(ab, dtype=int),
        "RCH": np.array(RCH, dtype=float),
        "WC_RCH": np.array(WC_RCH, dtype=float),
        "ycc": np.array(ycc, dtype=float),
        "zcc": np.array(zcc, dtype=float),
        "yaa": np.array(yaa, dtype=float),
        "zaa": np.array(zaa, dtype=float),
        "ybb": np.array(ybb, dtype=float),
        "zbb": np.array(zbb, dtype=float),
        "last_opt": last_opt,
    }


def main() -> None:
    result = run_simulation()

    print("Simulation complete.")
    print(f"Points count: {len(result['Ctz_plot'])}")
    print(f"Camber range (deg): {result['camber'].min():.6f} ~ {result['camber'].max():.6f}")
    print(f"Toe range (deg): {result['toe'].min():.6f} ~ {result['toe'].max():.6f}")
    print(f"RCH points: {len(result['RCH'])}")

    last_opt = result["last_opt"]
    if last_opt is not None:
        x = last_opt["x"]
        fvec = last_opt["fvec"]
        print("Final optimization result (Python):")
        print(f"  Ctz = {last_opt['Ctz']:.6f}")
        print(f"  ier = {last_opt['ier']}")
        print(f"  residual_norm = {last_opt['residual_norm']:.12e}")
        print(
            "  x = "
            f"[{x[0]:.12f}, {x[1]:.12f}, {x[2]:.12f}, {x[3]:.12f}, {x[4]:.12f}]"
        )
        print(
            "  residual fvec = "
            f"[{fvec[0]:.12e}, {fvec[1]:.12e}, {fvec[2]:.12e}, {fvec[3]:.12e}, {fvec[4]:.12e}]"
        )


if __name__ == "__main__":
    main()
