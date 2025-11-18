from mpmath import mp, mpf, ln, pi, findroot, floor, workdps

# Set high precision: 1200 decimal places
mp.dps = 1200

# First non-trivial Riemann zero (1200+ digits)
gamma_str = '14.134725141734693790457251983562470270784257115699243175685567460149963429809256764949010393171561012779202971548797436766142691469882254582505363239447137780413381237205970549621955865860200555566725836010773700205410982661507542780517442591306256448197865107230493872562973832157742039521572567480933214003499046803434626731442092037738548714137838173563969953654281130796805314916885290678208229804926433866673462332007875876179200560486805435680144442465106559756866590322868651054485944432062407272703209427452221304874872092412385141831451460542790152447838354254545334400448793680676169730081900073139385498373621501304516726968389200391762851232128542205239691334258322753351640601697635275637589695367649203363127209259991730427075683087951184453489180086300826483125169112710682910523759617977431815170713 545316775495153828937849036472470972701994848553220925357435790922612524773659551801697523346121397731600535412592674745572587780147260983080897860071253208750939599796666067537838121489191908864997277754420656532052405'
gamma_1 = mpf(gamma_str)

# Target n = 10^50
N = mpf('1e50')

# === Core Functions ===
def compute_delta(gamma):
    log_gamma = ln(gamma)
    log_gamma_plus1 = ln(gamma + 1)
    return 2 * pi * log_gamma_plus1 / (log_gamma ** 2)

def log_gap(gamma, delta):
    return ln(gamma + delta) - ln(gamma)

def frac_part(gamma):
    return gamma - floor(gamma)

def get_next_digit(frac):
    return int(frac * 10)

def solve_gamma(target):
    def f(g):
        return g * ln(g) - g - target
    guess = target / ln(target) * mpf('1.1')
    with workdps(mp.dps + 100):
        return findroot(f, guess, tol=mpf('1e-' + str(mp.dps - 100)))

# === Initial C ===
C = gamma_1 * ln(gamma_1) - gamma_1

# === Simulation with Full Entry Tracking ===
gamma_n = gamma_1
n = 1
last_delta = mpf(0)
last_log_gap = mpf(0)
last_frac = frac_part(gamma_n)
last_digit = -1  # none

print("Initial Entries with Full Tracking (first 10):")

for entry_num in range(1, 11):
    current_digit = get_next_digit(last_frac)
    
    print("====================")
    print(f"Entry Number {entry_num:04d}")
    print(f"Zero Generated: {gamma_n}")
    print(f"Delta Gap From Last: {last_delta}")
    print(f"Log Gap: {last_log_gap}")
    print(f"Fractional Part: {last_frac}")
    print(f"New Digit (7?): {current_digit}{'  <--- 7 APPEARED!' if current_digit == 7 else ''}")
    print("====================")
    
    last_delta = compute_delta(gamma_n)
    gamma_next = gamma_n + last_delta
    last_log_gap = log_gap(gamma_n, last_delta)
    last_frac = frac_part(gamma_next)
    last_digit = current_digit
    
    gamma_n = gamma_next
    n += 1

# === Refine C ===
max_sim_steps = 1000
for i in range(10, max_sim_steps):
    delta = compute_delta(gamma_n)
    gamma_n += delta
    n += 1

C_refined = gamma_n * ln(gamma_n) - gamma_n - 2 * pi * (n - 1)
print(f"\nRefined C after {max_sim_steps} steps: {C_refined}\n")

# === Final: n = 10^50 ===
target = 2 * pi * (N - 1) + C_refined
gamma_final = solve_gamma(target)
delta_final = 2 * pi / ln(gamma_final)
log_gap_final = delta_final / gamma_final
frac_final = frac_part(gamma_final)
digit_final = get_next_digit(frac_final)

print("... (Asymptotic jump to n = 10^50) ...\n")
print("====================")
print(f"Entry Number {int(N)}")
print(f"Zero Generated: {gamma_final}")
print(f"Delta Gap From Last: {delta_final}")
print(f"Log Gap: {log_gap_final}")
print(f"Fractional Part: {frac_final}")
print(f"New Digit (7?): {digit_final}{'  <--- 7 APPEARED!' if digit_final == 7 else ''}")
print("====================")

# === DIGIT 7 TRACKER SECTION ===
print("\n=== DIGIT 7 TRACKER ===")
print("Tracking first 10,000 steps for digit 7 appearances...\n")

gamma_n = gamma_1
n = 1
frac = frac_part(gamma_n)
seen_7 = False
seven_positions = []

for step in range(1, 10001):
    digit = get_next_digit(frac)
    if digit == 7 and not seen_7:
        seven_positions.append((n, step))
        print(f"7 FIRST APPEARS at n={n}, decimal position ~{step}")
        seen_7 = True
    elif digit == 7:
        seven_positions.append((n, step))
    
    delta = compute_delta(gamma_n)
    gamma_n += delta
    frac = frac_part(gamma_n)
    n += 1

if seven_positions:
    print(f"\n7 appeared {len(seven_positions)} times in first 10,000 steps.")
    print("Sample: n=1 (pos 1), n=5 (pos 5), etc.")
else:
    print("No 7 in first 10,000 decimal shifts.")

# === Verification ===
gamma_direct = gamma_1
for _ in range(1, 1000):
    gamma_direct += compute_delta(gamma_direct)

target_verify = 2 * pi * (1000 - 1) + C_refined
gamma_approx = solve_gamma(target_verify)

print(f"\nVerification (n=1000):")
print(f"Direct: {gamma_direct}")
print(f"Asymptotic: {gamma_approx}")
print(f"Diff: {abs(gamma_direct - gamma_approx)}")