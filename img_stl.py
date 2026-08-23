import struct

import numpy as np
from PIL import Image


def make_lithophane_stl(
    image_path: str,
    output_stl_path: str,
    width_mm: float = 150.0,
    height_mm: float = 150.0,
    min_thick: float = 0.8,
    max_thick: float = 3.2,
    res: int = 300,
):
    """Generates a closed, manifold binary STL lithophane plate from an image."""

    #Load image and crop to square 1:1 aspect ratio
    img = Image.open(image_path).convert("L")
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((res, res), Image.Resampling.LANCZOS)

    # Normalize image brightness and convert to Z-heights
    # White pixels will be min_thick and the black pixels shoulb be max_thick
    img_arr = np.array(img, dtype=np.float32) / 255.0
    z_top = max_thick - (img_arr * (max_thick - min_thick))

    # Mapping the X, Y coordinate grids
    x = np.linspace(0, width_mm, res)
    y = np.linspace(0, height_mm, res)
    xx, yy = np.meshgrid(x, y)

    triangles = []

    def calc_normal(p1, p2, p3):
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        return n / norm if norm > 0 else np.array([0.0, 0.0, 0.0])

    # Build Top Heightmap Surface
    for r in range(res - 1):
        for c in range(res - 1):
            p00 = np.array([xx[r, c], yy[r, c], z_top[r, c]])
            p10 = np.array([xx[r + 1, c], yy[r + 1, c], z_top[r + 1, c]])
            p01 = np.array([xx[r, c + 1], yy[r, c + 1], z_top[r, c + 1]])
            p11 = np.array([xx[r + 1, c + 1], yy[r + 1, c + 1], z_top[r + 1, c + 1]])

            triangles.append((p00, p01, p10))
            triangles.append((p10, p01, p11))

    # Build Flat Bottom Base
    for r in range(res - 1):
        for c in range(res - 1):
            b00 = np.array([xx[r, c], yy[r, c], 0.0])
            b10 = np.array([xx[r + 1, c], yy[r + 1, c], 0.0])
            b01 = np.array([xx[r, c + 1], yy[r, c + 1], 0.0])
            b11 = np.array([xx[r + 1, c + 1], yy[r + 1, c + 1], 0.0])

            triangles.append((b00, b10, b01))
            triangles.append((b10, b11, b01))

    # Build Side Walls
    # Bottom Edge
    # y=0
    for c in range(res - 1):
        t0, t1 = (
            np.array([xx[0, c], yy[0, c], z_top[0, c]]),
            np.array([xx[0, c + 1], yy[0, c + 1], z_top[0, c + 1]]),
        )
        b0, b1 = (
            np.array([xx[0, c], yy[0, c], 0.0]),
            np.array([xx[0, c + 1], yy[0, c + 1], 0.0]),
        )
        triangles.append((b0, t1, t0))
        triangles.append((b0, b1, t1))

    # Top Edge
    # y=ymax
    for c in range(res - 1):
        t0, t1 = (
            np.array([xx[res - 1, c], yy[res - 1, c], z_top[res - 1, c]]),
            np.array(
                [
                    xx[res - 1, c + 1],
                    yy[res - 1, c + 1],
                    z_top[res - 1, c + 1],
                ]
            ),
        )
        b0, b1 = (
            np.array([xx[res - 1, c], yy[res - 1, c], 0.0]),
            np.array([xx[res - 1, c + 1], yy[res - 1, c + 1], 0.0]),
        )
        triangles.append((b0, t0, t1))
        triangles.append((b0, t1, b1))

    # Left Edge
    # x=0
    for r in range(res - 1):
        t0, t1 = (
            np.array([xx[r, 0], yy[r, 0], z_top[r, 0]]),
            np.array([xx[r + 1, 0], yy[r + 1, 0], z_top[r + 1, 0]]),
        )
        b0, b1 = (
            np.array([xx[r, 0], yy[r, 0], 0.0]),
            np.array([xx[r + 1, 0], yy[r + 1, 0], 0.0]),
        )
        triangles.append((b0, t0, t1))
        triangles.append((b0, t1, b1))

    # Right Edge
    # x=max
    for r in range(res - 1):
        t0, t1 = (
            np.array([xx[r, res - 1], yy[r, res - 1], z_top[r, res - 1]]),
            np.array(
                [
                    xx[r + 1, res - 1],
                    yy[r + 1, res - 1],
                    z_top[r + 1, res - 1],
                ]
            ),
        )
        b0, b1 = (
            np.array([xx[r, res - 1], yy[r, res - 1], 0.0]),
            np.array([xx[r + 1, res - 1], yy[r + 1, res - 1], 0.0]),
        )
        triangles.append((b0, t1, t0))
        triangles.append((b0, b1, t1))

    # Export Binary STL File
    with open(output_stl_path, "wb") as f:
        header = b"Lithophane Plate STL".ljust(80, b"\x00")
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))

        for p1, p2, p3 in triangles:
            n = calc_normal(p1, p2, p3)
            f.write(
                struct.pack(
                    "<12fH",
                    n[0],
                    n[1],
                    n[2],
                    p1[0],
                    p1[1],
                    p1[2],
                    p2[0],
                    p2[1],
                    p2[2],
                    p3[0],
                    p3[1],
                    p3[2],
                    0,
                )
            )

    print(f"STL generated successfully: {output_stl_path}")

if __name__ == "__main__":
    make_lithophane_stl(
        image_path="example_image.jpeg",
        output_stl_path="example_lithophane_150x150.stl",
        width_mm=150.0,
        height_mm=150.0,
        min_thick=0.2,
        max_thick=1.2,
        res=500,
    )
