#!/usr/bin/env python3

"""
Riemann Zero Generator
----------------------
Sequentially computes all Riemann zeros (trivial and non-trivial),
outputs each one with high precision (1200 decimal places),
and logs every non-trivial zero with a clear notification.

Features:
- Trivial zeros: exact, s = -2, -4, -6, ...
- Non-trivial zeros: found numerically via sign changes in Re[ζ(1/2 + it)]
- Precision: 1200 decimal places
- Measures decay (gap) between consecutive non-trivial zeros
- Real-time notification on non-trivial zero discovery
- Clean, server-friendly output
- For testing, prints shortened (10 digits); set print_precision=1200 for full
- Runs until 10^50 non-trivial zeros (theoretically; interrupt in practice)
"""

import mpmath as mp
from mpmath import mpf, mpc, zeta, findroot

# Set precision to 1200 decimal places
mp.dps = 1200
mp.pretty = False

# For output: shorten for test; set to 1200 for full precision print
print_precision = 10  # Increase to 1200 for full output on server


class RiemannZeroStream:
    def __init__(self):
        self.trivial_index = 1
        self.last_t = mpf(0)          # Last scanned imaginary part for search
        self.zero_count = 0           # Total zeros emitted
        self.non_trivial_count = 0
        self.last_non_trivial_t = None  # For measuring gaps between non-trivial

    def _zeta_real(self, t):
        """Real part of ζ(0.5 + i t)"""
        return zeta(mpc(0.5, t)).real

    def _find_next_non_trivial(self):
        """Locate next non-trivial zero above self.last_t"""
        t = self.last_t
        step = mpf('0.05')
        max_steps = 100000

        z_prev = self._zeta_real(t)
        for _ in range(max_steps):
            t += step
            z_curr = self._zeta_real(t)
            if z_prev * z_curr < 0:  # Sign change → zero in [t-step, t]
                # Refine with secant method
                root = findroot(self._zeta_real, (t - step, t), solver='secant', tol=mpf('1e-1200'))
                self.last_t = root + mpf('1e-10')  # Skip past this zero
                return root
            z_prev = z_curr
        raise RuntimeError("No zero found in search range")

    def generate_next(self):
        """
        Yield the next Riemann zero (trivial or non-trivial).
        Emits in approximate order by increasing magnitude.
        """
        # Next trivial zero: s = -2 * n
        s_trivial = mpf(-2 * self.trivial_index)

        # Estimate next non-trivial zero height (asymptotic for large n)
        if self.non_trivial_count == 0:
            next_non_trivial_t = mpf('14.13')
        else:
            n = self.non_trivial_count + 1
            next_non_trivial_t = (2 * mp.pi * n) / mp.log(n / (2 * mp.pi)) + mpf('10')

        # Decide which comes first (compare |trivial| < next t)
        if abs(s_trivial) < next_non_trivial_t:
            # Trivial zero comes next
            rho = mpc(s_trivial, 0)
            self.trivial_index += 1
            self.zero_count += 1
            return rho, 'trivial'
        else:
            # Find next non-trivial zero
            t = self._find_next_non_trivial()
            rho = mpc(0.5, t)
            # Compute decay (gap from previous non-trivial)
            if self.last_non_trivial_t is not None:
                gap = t - self.last_non_trivial_t
            else:
                gap = None
            self.last_non_trivial_t = t
            self.non_trivial_count += 1
            self.zero_count += 1
            return rho, 'non-trivial', gap

    def stream(self):
        """Generator: yields (zero, type, gap) indefinitely"""
        print("RIEMANN ZERO STREAM STARTED")
        print("=" * 80)
        while True:
            rho, kind, *gap = self.generate_next()  # gap is list or empty
            gap = gap[0] if gap else None
            if rho.imag == 0:
                s_str = mp.nstr(rho.real, print_precision)
            else:
                s_str = f"{mp.nstr(rho.real, print_precision)}+{mp.nstr(rho.imag, print_precision)}j"
            if kind == 'non-trivial':
                print(f"\nNON-TRIVIAL ZERO #{self.non_trivial_count} FOUND:")
                print(f"   ρ = {s_str}")
                print(f"   |ρ| ≈ {abs(rho):.6f}")
                if gap is not None:
                    print(f"   Decay (gap from previous non-trivial): {mp.nstr(gap, print_precision)}")
                print(f"   Total zeros so far: {self.zero_count}")
                print("-" * 80)
            else:
                print(f"Trivial zero: {s_str}  (total: {self.zero_count})")
            yield rho, kind, gap


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    finder = RiemannZeroStream()
    stream = finder.stream()
    try:
        while finder.non_trivial_count < 10**50:
            next(stream)
    except KeyboardInterrupt:
        print("\n\nStream interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")