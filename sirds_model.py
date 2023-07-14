import numpy as np
from scipy.integrate import solve_ivp

TOTAL_POPULATION = 100000

def sirds_model_kernel(t, y, beta0, gamma0, f0, l0, p1=None, beta1=None, gamma1=None, f1=None, l1=None, p2=None, beta2=None, gamma2=None, f2=None, l2=None, p3=None, beta3=None, gamma3=None, f3=None, l3=None, p_inner_0=None, beta_inner_0=None):
    S, I, R, D, I_accumulated = y
    N = S+I+R+D

    proportion_S = S/N
    proportion_I = I/N
    proportion_R = R/N

    beta = beta0
    gamma = gamma0
    f = f0
    l = l0

    if (p_inner_0 is not None) and (t >= p_inner_0):
        beta = beta_inner_0

    if (p1 is not None) and (t >= p1):
        beta = beta1
        gamma = gamma1
        f = f1
        l = l1

    if (p2 is not None) and (t >= p2):
        beta = beta2
        gamma = gamma2
        f = f2
        l = l2

    if (p3 is not None) and (t >= p3):
        beta = beta3
        gamma = gamma3
        f = f3
        l = l3

    dSdt = (-beta*proportion_S*proportion_I) + (l*proportion_R)
    dIdt = (beta*proportion_S*proportion_I) - (gamma*proportion_I)
    dI_accumulated_dt = (beta * proportion_S * proportion_I)
    dRdt = (1-f) * (gamma*proportion_I) - (l*proportion_R)
    dDdt = gamma*f*proportion_I

    return dSdt * N, dIdt * N, dRdt * N, dDdt * N, dI_accumulated_dt * N

def brazil_covid_sirds_model(
        initial_infected_population,
        period_in_days,
        days_between_infections_0,
        days_to_recovery_0,
        case_fatality_probability_0,
        loss_immunity_in_days_0,
        breakpoint_1=None,
        days_between_infections_1=None,
        days_to_recovery_1=None,
        case_fatality_probability_1=None,
        loss_immunity_in_days_1=None,
        breakpoint_2=None,
        days_between_infections_2=None,
        days_to_recovery_2=None,
        case_fatality_probability_2=None,
        loss_immunity_in_days_2=None,
        breakpoint_3=None,
        days_between_infections_3=None,
        days_to_recovery_3=None,
        case_fatality_probability_3=None,
        loss_immunity_in_days_3=None,
        total_population = TOTAL_POPULATION,
        wave_inner_breakpoint_1=None,
        days_between_infections_inner_wave_1=None):

    I0 = initial_infected_population
    S0 = total_population - I0
    R0 = 0
    D0 = 0

    # Initial conditions vector
    y0 = S0, I0, R0, D0, I0
    # A grid of time points (in days)
    t = np.linspace(0, int(period_in_days-1), int(period_in_days), dtype=int)

    # Contact rate, beta, (in 1/days).
    beta0 = 1. / days_between_infections_0
    # Recovery rate, gamma, (in 1/days).
    gamma0 = 1./days_to_recovery_0
    # Daily lethality rate, f0, (in 1/days).
    f0 = case_fatality_probability_0
    # Loss immunity rate, l0  (in 1/days).
    l0 = 1./loss_immunity_in_days_0

    # Change procated by lockdown in first wave
    if (wave_inner_breakpoint_1 is not None):
        p_inner_0 = wave_inner_breakpoint_1
        beta_inner_0 = 1. / days_between_infections_inner_wave_1
    else:
        p_inner_0 = None
        beta_inner_0 = None

    # The first turning point in pandemic (begin second wave)
    if (breakpoint_1 is not None):
        p1 = breakpoint_1
        beta1 = 1. / days_between_infections_1
        gamma1 = 1. / days_to_recovery_1
        f1 = case_fatality_probability_1
        l1 = 1./loss_immunity_in_days_1
    else:
        p1 = None
        beta1 = None
        gamma1 = None
        f1 = None
        l1 = None

    # The second turning point in pandemic (begin third wave)
    if (breakpoint_2 is not None):
        p2 = breakpoint_2
        beta2 = 1. / days_between_infections_2
        gamma2 = 1./ days_to_recovery_2
        f2 = case_fatality_probability_2
        l2 = 1./loss_immunity_in_days_2
    else:
        p2 = None
        beta2 = None
        gamma2 = None
        f2 = None
        l2 = None

    # The third turning point in pandemic (begin fourth wave)
    if (breakpoint_3 is not None):
        p3 = breakpoint_3
        beta3 = 1. / days_between_infections_3
        gamma3 = 1./ days_to_recovery_3
        f3 = case_fatality_probability_3
        l3 = 1./loss_immunity_in_days_3
    else:
        p3 = None
        beta3 = None
        gamma3 = None
        f3 = None
        l3 = None

    ret = solve_ivp(
        sirds_model_kernel,
        [t.min(), t.max()],
        y0,
        t_eval=t,
        args=(beta0, gamma0, f0, l0, p1, beta1, gamma1, f1, l1, p2, beta2, gamma2, f2, l2, p3, beta3, gamma3, f3, l3, p_inner_0, beta_inner_0),
        method='LSODA'
    )
    S, I, R, D, I_accumulated = ret.y

    return S, I, R, D, I_accumulated

def get_new_deaths(D):
    D_new_deaths = D[1:] - D[:len(D)-1]
    return D_new_deaths

def get_effective_reproduction_number(S, N, days_to_recovery_0, days_between_infections_0, breakpoint_1, days_to_recovery_1, days_between_infections_1, breakpoint_2, days_to_recovery_2, days_between_infections_2, breakpoint_3, days_to_recovery_3, days_between_infections_3, wave_inner_breakpoint_1, days_between_infections_inner_wave_1):
    R0 = days_to_recovery_0 / days_between_infections_0
    if days_between_infections_inner_wave_1 is not None:
        Rp_inner_0 = days_to_recovery_0 / days_between_infections_inner_wave_1
    Rp1 = days_to_recovery_1 / days_between_infections_1
    Rp2 = days_to_recovery_2 / days_between_infections_2
    Rp3 = days_to_recovery_3 / days_between_infections_3

    Rt = np.array([R0])

    for t in range(1,len(S)):
        Rp = R0
        if (wave_inner_breakpoint_1 is not None) and (t >= wave_inner_breakpoint_1):
            Rp = Rp_inner_0
        if t >= breakpoint_1:
            Rp = Rp1
        if t >= breakpoint_2:
            Rp = Rp2
        if t >= breakpoint_3:
            Rp = Rp3
        reproductive_number = Rp * (S[t-1]/N)
        Rt = np.append(Rt, reproductive_number)

    return Rt

def get_new_cases(I_accumulated):
    I_new_cases = I_accumulated[1:] - I_accumulated[:len(I_accumulated)-1]
    return I_new_cases