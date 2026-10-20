"""Independent arithmetic on T437's published rounded summaries; no producer imports."""
import math

for name, n, charge in (("D1", 2.15e6, 1.449e-8), ("D2", 1.81e6, 1.240e-8)):
    projected = n * (1e-10 / charge) ** (1 / -.46)
    print(name, "projected_boxes", projected, "CPU_hours_at_recorded_throughput", projected / n)
print("root/slab", 2.65e-5 / 5.5975e-8)
print("D1,D2 untilted/tilted", 1.464 / 1.449, 1.242 / 1.240)
print("fraction-derived gains (estimates only)", *(1 / f for f in (.998, .82, .94)))
print("D2/slab", 1.24e-8 / 5.5975e-8)
print("sufficient conditional probability", 1e-10 / 1.24e-8)
print("domain tail", math.erfc(10 / math.sqrt(2)) + 4 * math.erfc(5))
print("primary endpoint slope", math.log(5.45e-7 / 1.88e-7) / math.log(3.74e6 / 4.5e5))
