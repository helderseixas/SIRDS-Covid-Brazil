# Baud, David, et al. "Real estimates of mortality following COVID-19 infection." The Lancet infectious diseases 20.7 (2020): 773.
from datetime import timedelta

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.integrate import odeint, solve_ivp
from sklearn.metrics import mean_absolute_error

from skfuzzy.control.fuzzyvariable import FuzzyVariable
import skfuzzy as fuzz

TOTAL_POPULATION = 100000

# Pulliam, Juliet RC, et al. "Increased risk of SARS-CoV-2 reinfection associated with emergence of Omicron in South Africa." Science 376.6593 (2022): eabn4947.
# This work report reinfection cases 90 days after first infection. This work also show reinfection peaks 170, 350, and 520 days after last infection.
DEFAULT_LOSS_IMMUNITY_IN_DAYS = 180
MIN_LOSS_IMMUNITY_IN_DAYS = 90
MAX_LOSS_IMMUNITY_IN_DAYS = 365
MIN_TRANSITION_DAYS_BETWEEN_EPIDEIC_PERIODS = 0
MAX_TRANSITION_DAYS_BETWEEN_EPIDEIC_PERIODS = 56
DEFAULT_TRANSITION_DAYS_BETWEEN_EPIDEMIC_PERIODS = 16
MIN_INTERVAL_BETWEEN_EPIDEMIC_PERIOD_WITH_SLOW_TRANSITION = 180

#Marra, Valerio, and Miguel Quartin. "A Bayesian estimate of the early COVID-19 infection fatality ratio in Brazil based on a random seroprevalence survey." International Journal of Infectious Diseases 111 (2021): 190-195.
REFERENCE_CASE_FATALITY_RATE = 0.0103

MIN_SIMILARITY_BETWEEN_SIDES_BELL_CURVE = -0.39
MAX_SIMILARITY_BETWEEN_SIDES_BELL_CURVE = 0.22
MIN_SIMILARITY_DTW = 0.22

BOUND_NOISE = 0.000001

def epidemic_parameter_defuzzification(epidemic_periods_fuzzy_variable, epidemic_parameters_crisp_values, specif_day_in_time_series):
    membership_values = np.array([])
    for term_name, term in epidemic_periods_fuzzy_variable.terms.items():
        membership_value = fuzz.interp_membership(epidemic_periods_fuzzy_variable.universe, term.mf, int(specif_day_in_time_series))
        membership_values = np.append(membership_values, [membership_value])

    return (membership_values * epidemic_parameters_crisp_values).sum() / membership_values.sum()

# Gallos, Lazaros K., and Nina H. Fefferman. "The effect of disease-induced mortality on structural network properties." PloS one 10.8 (2015): e0136704.
def sirds_model_kernel(t, y, *params):
    # Initialize values
    S, I, R, D, I_accumulated = y
    N = S + I + R + D
    proportion_S = S / N
    proportion_I = I / N
    proportion_R = R / N

    beta = params[0]
    gamma = params[1]
    f = params[2]
    l = params[3]

    if len(params) > 4:
        # Extract parameters for each breakpoint
        params_breakpoints = params[4:]
        num_breakpoints = int(len(params_breakpoints) / 5)

        beta_values = params_breakpoints[:num_breakpoints]
        gamma_values = params_breakpoints[num_breakpoints:2 * num_breakpoints]
        f_values = params_breakpoints[2 * num_breakpoints:3 * num_breakpoints]
        l_values = params_breakpoints[3 * num_breakpoints:4 * num_breakpoints]
        breakpoint_values = params_breakpoints[4 * num_breakpoints:]

        # Determine the correct parameter values based on the current time
        for i in range(num_breakpoints):
            if t >= breakpoint_values[i]:
                beta = beta_values[i]
                gamma = gamma_values[i]
                f = f_values[i]
                l = l_values[i]

    dSdt = (-beta * proportion_S * proportion_I) + (l * proportion_R)
    dIdt = (beta * proportion_S * proportion_I) - (gamma * proportion_I)
    dI_accumulated_dt = (beta * proportion_S * proportion_I)
    dRdt = (1 - f) * (gamma * proportion_I) - (l * proportion_R)
    dDdt = gamma * f * proportion_I

    return dSdt * N, dIdt * N, dRdt * N, dDdt * N, dI_accumulated_dt * N

def sirds_model_fuzzy_kernel(t, y, *params):
    # Initialize values
    S, I, R, D, I_accumulated = y
    N = S + I + R + D
    proportion_S = S / N
    proportion_I = I / N
    proportion_R = R / N

    beta_fuzzy_variable, beta_values = params[0]
    gamma = params[1]
    f_fuzzy_variable, f_values = params[2]
    l_fuzzy_variable, l_values = params[3]

    # Fuzzy parameters
    beta = epidemic_parameter_defuzzification(beta_fuzzy_variable, beta_values, t)
    f = epidemic_parameter_defuzzification(f_fuzzy_variable, f_values, t)
    l = epidemic_parameter_defuzzification(l_fuzzy_variable, l_values, t)

    dSdt = (-beta * proportion_S * proportion_I) + (l * proportion_R)
    dIdt = (beta * proportion_S * proportion_I) - (gamma * proportion_I)
    dI_accumulated_dt = (beta * proportion_S * proportion_I)
    dRdt = (1 - f) * (gamma * proportion_I) - (l * proportion_R)
    dDdt = gamma * f * proportion_I

    return dSdt * N, dIdt * N, dRdt * N, dDdt * N, dI_accumulated_dt * N

def get_epidemic_periods_with_slow_transition_fuzzy_variable(period_in_days, breakpoint_values):
    fuzzy_variable = FuzzyVariable(np.linspace(0, int(period_in_days - 1), int(period_in_days)), 'time')

    if len(breakpoint_values) > 0:
        fuzzy_variable['0'] = fuzz.trimf(fuzzy_variable.universe,[0, 0, breakpoint_values[0]])

        for i in range(len(breakpoint_values)):
            if i == 0:
                a = 0
            else:
                a = breakpoint_values[i-1]
            b = breakpoint_values[i]
            if i < len(breakpoint_values)-1:
                c = breakpoint_values[i+1]
                fuzzy_variable[str(i + 1)] = fuzz.trimf(fuzzy_variable.universe, [a,b,c])
            else:
                c = period_in_days
                d = period_in_days
                fuzzy_variable[str(i + 1)] = fuzz.trapmf(fuzzy_variable.universe, [a,b,c,d])
    else:
        fuzzy_variable['0'] = fuzz.trapmf(fuzzy_variable.universe,
                                                [0, 0, period_in_days, period_in_days])
    return fuzzy_variable

def get_epidemic_periods_for_beta_transition_fuzzy_variable(period_in_days, breakpoint_values, transition_days_values):
    variable_trapezoidal = FuzzyVariable(np.linspace(0, int(period_in_days - 1), int(period_in_days)), 'time')
    if len(breakpoint_values) > 0:
        variable_trapezoidal['0'] = fuzz.trapmf(variable_trapezoidal.universe,
                                                [0, 0, breakpoint_values[0], breakpoint_values[0] + round(transition_days_values[0])])

        for i in range(len(breakpoint_values)):
            a = breakpoint_values[i] - round(transition_days_values[i])
            b = breakpoint_values[i]
            index_c = i + 1
            if index_c < len(breakpoint_values):
                c = breakpoint_values[index_c]
                d = c + round(transition_days_values[index_c])
            else:
                c = period_in_days
                d = c

            sort_breakpoints = sorted([a, b, c, d])
            variable_trapezoidal[str(i + 1)] = fuzz.trapmf(variable_trapezoidal.universe, sort_breakpoints)
    else:
        variable_trapezoidal['0'] = fuzz.trapmf(variable_trapezoidal.universe,
                                                [0, 0, period_in_days, period_in_days])

    return variable_trapezoidal

def get_epidemic_periods_with_crisp_transition_fuzzy_variable(period_in_days, breakpoint_values):
    variable_trapezoidal = FuzzyVariable(np.linspace(0, int(period_in_days - 1), int(period_in_days)), 'time')
    variable_trapezoidal['0'] = fuzz.trapmf(variable_trapezoidal.universe,
                                            [0, 0, breakpoint_values[0], breakpoint_values[0]])

    for i in range(len(breakpoint_values)):
        a = breakpoint_values[i]
        b = breakpoint_values[i]
        index_c = i + 1
        if index_c >= len(breakpoint_values):
            c = period_in_days
        else:
            c = breakpoint_values[index_c]
        d = c
        variable_trapezoidal[str(i + 1)] = fuzz.trapmf(variable_trapezoidal.universe, [a, b, c, d])

    return variable_trapezoidal

def sirds_model_fuzzy(parameters):
    I0 = parameters[0]
    S0 = TOTAL_POPULATION - I0
    R0 = 0
    D0 = 0

    # Initial conditions vector
    y0 = S0, I0, R0, D0, I0

    # A grid of time points (in days)
    period_in_days = parameters[1]
    t = np.linspace(0, int(period_in_days - 1), int(period_in_days), dtype=int)

    #Initial SIRDS parameters
    days_between_infections = parameters[2]
    days_to_recovery = parameters[3]

    list_breakpoints_in_slow_transition = parameters[4]

    beta = 1.0 / days_between_infections
    gamma = 1.0 / days_to_recovery

    quantity_epidemic_periods_with_slow_transition = len(list_breakpoints_in_slow_transition) + 1
    f_values = parameters[5:5+quantity_epidemic_periods_with_slow_transition]
    l_values = [1.0 / loss_immunity_in_days for loss_immunity_in_days in
                parameters[5+quantity_epidemic_periods_with_slow_transition:5+2*quantity_epidemic_periods_with_slow_transition]]

    num_breakpoints = int(len(parameters[5 + 2 * quantity_epidemic_periods_with_slow_transition:])/3)

    # Breakpoint parameters
    if num_breakpoints > 0:
        beta_values = [1.0 / days_between_infections for days_between_infections in parameters[5+2*quantity_epidemic_periods_with_slow_transition:5+2*quantity_epidemic_periods_with_slow_transition+num_breakpoints]]
        beta_breakpoint_values = parameters[5+2*quantity_epidemic_periods_with_slow_transition+num_breakpoints : 5+2*quantity_epidemic_periods_with_slow_transition+2*num_breakpoints]
        transition_days_between_epidemic_periods_values = parameters[5+2*quantity_epidemic_periods_with_slow_transition+2*num_breakpoints:]
    else:
        beta_values = []
        beta_breakpoint_values = []
        transition_days_between_epidemic_periods_values = []

    slow_transition_breakpoint_values = []
    for indice_breakpoint in list_breakpoints_in_slow_transition:
        slow_transition_breakpoint_values.append(beta_breakpoint_values[indice_breakpoint])

    beta_values = np.append([beta], beta_values)
    periods_with_fast_transition = get_epidemic_periods_for_beta_transition_fuzzy_variable(period_in_days,
                                                                                           beta_breakpoint_values,
                                                                                           transition_days_between_epidemic_periods_values)
    periods_with_slow_transition = get_epidemic_periods_with_slow_transition_fuzzy_variable(period_in_days, slow_transition_breakpoint_values)

    parameter_values = [(periods_with_fast_transition, beta_values),
                      gamma,
                      (periods_with_slow_transition, f_values),
                      (periods_with_slow_transition, l_values)]

    # Solve the differential equation
    ret = solve_ivp(
        sirds_model_fuzzy_kernel,
        [t.min(), t.max()],
        y0,
        t_eval=t,
        args=tuple([*parameter_values]),
        method='LSODA'
    )
    S, I, R, D, I_accumulated = ret.y

    return S, I, R, D, I_accumulated

def sirds_model(parameters):
    I0 = parameters[0]
    S0 = TOTAL_POPULATION - I0
    R0 = 0
    D0 = 0

    # Initial conditions vector
    y0 = S0, I0, R0, D0, I0

    # A grid of time points (in days)
    period_in_days = parameters[1]
    t = np.linspace(0, int(period_in_days - 1), int(period_in_days), dtype=int)

    #Initial SIRDS parameters
    days_between_infections = parameters[2]
    days_to_recovery = parameters[3]
    case_fatality_probability = parameters[4]
    loss_immunity_in_days = parameters[5]

    beta = 1.0 / days_between_infections
    gamma = 1.0 / days_to_recovery
    f = case_fatality_probability
    l = 1.0 / loss_immunity_in_days
    initial_values = [beta, gamma, f, l]

    # Breakpoint parameters
    if len(parameters) > 6:
        # Extract parameters for each breakpoint
        params_breakpoints = parameters[6:]
        num_breakpoints = int(len(params_breakpoints) / 5)

        beta_values = [1.0 / days_between_infections for days_between_infections in params_breakpoints[:num_breakpoints]]
        gamma_values = [1.0 / days_to_recovery for days_to_recovery in
                        params_breakpoints[num_breakpoints:2 * num_breakpoints]]
        f_values = params_breakpoints[2 * num_breakpoints:3 * num_breakpoints]
        l_values = [1.0 / loss_immunity_in_days for loss_immunity_in_days in params_breakpoints[3 * num_breakpoints:4 * num_breakpoints]]
        breakpoint_values = params_breakpoints[4 * num_breakpoints:]
    else:
        beta_values = []
        gamma_values = []
        f_values = []
        l_values = []
        breakpoint_values = []

    # Solve the differential equation
    ret = solve_ivp(
        sirds_model_kernel,
        [t.min(), t.max()],
        y0,
        t_eval=t,
        args=tuple([*initial_values, *beta_values, *gamma_values, *f_values, *l_values, *breakpoint_values]),
        method='LSODA'
    )
    S, I, R, D, I_accumulated = ret.y

    return S, I, R, D, I_accumulated

def need_adjust_first_outbreak(effective_reproduction_number_in_first_outbreak,
                               date_first_case,
                               date_end_outbreak,
                               initial_infected_population,
                               days_between_infections,
                               days_to_recovery,
                               case_fatality_rate,
                               loss_immunity_in_days):
    r0 = np.max(effective_reproduction_number_in_first_outbreak)
    estimated_days_between_infections = days_to_recovery / r0

    period = (date_end_outbreak - date_first_case).days + 1
    list_breakpoints_in_slow_transition = []

    y =  sirds_model_fuzzy(tuple([*[
        initial_infected_population,
        period,
        estimated_days_between_infections,
        days_to_recovery, list_breakpoints_in_slow_transition, case_fatality_rate, loss_immunity_in_days]]))
    S, I, R, D, I_accumulated = y

    epidemic_periods_with_fast_transition_fuzzy_variable = (
        get_epidemic_periods_for_beta_transition_fuzzy_variable(period, [], []))

    SIRDS_effective_reproduction_number = (
        get_fuzzy_effective_reproduction_number(S,
                                                TOTAL_POPULATION,
                                                tuple([*[estimated_days_between_infections,
    (epidemic_periods_with_fast_transition_fuzzy_variable, [days_between_infections])]])))

    distance, path = fastdtw(effective_reproduction_number_in_first_outbreak, SIRDS_effective_reproduction_number)

    if distance > MIN_SIMILARITY_DTW:
        return True
    else:
        return False


def similarity_between_sides_in_bell_curve(time_series):
    peak_index = np.argmax(time_series)
    sum_left_side = np.sum(time_series[:peak_index])
    sum_right_side = np.sum(time_series[peak_index + 1:])

    return (sum_right_side - sum_left_side) / (sum_right_side + sum_left_side)

def need_adjust_secondary_outbreak(effective_reproduction_number_in_outbreak):
    similarity = similarity_between_sides_in_bell_curve(effective_reproduction_number_in_outbreak)

    if (similarity >= MIN_SIMILARITY_BETWEEN_SIDES_BELL_CURVE) & (similarity <= MAX_SIMILARITY_BETWEEN_SIDES_BELL_CURVE):
        return False
    else:
        return True

def get_new_deaths(D):
    D_new_deaths = D[1:] - D[:len(D)-1]
    return D_new_deaths

def get_effective_reproduction_number(S, N, parameters):
    days_to_recovery = parameters[0]
    days_between_infections = parameters[1]
    R0 = days_to_recovery / days_between_infections
    Rt = np.array([R0])

    if len(parameters) > 2:
        # Extract parameters for each breakpoint
        params_breakpoints = parameters[2:]
        num_breakpoints = int(len(params_breakpoints) / 3)

        days_to_recovery_values = params_breakpoints[:num_breakpoints]
        days_between_infections_values = params_breakpoints[num_breakpoints:2 * num_breakpoints]
        breakpoint_values = params_breakpoints[2 * num_breakpoints:]

    for t in range(1,len(S)):
        Rp = R0
        for i in range(num_breakpoints):
            if (len(parameters) > 2) & (t >= breakpoint_values[i]):
                Rp = days_to_recovery_values[i] / days_between_infections_values[i]

        reproductive_number = Rp * (S[t-1]/N)
        Rt = np.append(Rt, reproductive_number)

    return Rt

def get_fuzzy_effective_reproduction_number(S, N, parameters):
    days_to_recovery = parameters[0]

    days_between_infections_fuzzy_variable, days_between_infections_values = parameters[1]
    days_between_infections = epidemic_parameter_defuzzification(days_between_infections_fuzzy_variable, days_between_infections_values, 0)

    R0 = days_to_recovery / days_between_infections
    Rt = np.array([R0])

    for t in range(1,len(S)):
        days_between_infections = epidemic_parameter_defuzzification(days_between_infections_fuzzy_variable, days_between_infections_values, t)

        Rp = days_to_recovery / days_between_infections

        reproductive_number = Rp * (S[t-1]/N)
        Rt = np.append(Rt, reproductive_number)

    return Rt

def get_new_cases(I_accumulated):
    I_new_cases = I_accumulated[1:] - I_accumulated[:len(I_accumulated)-1]
    return I_new_cases

def create_bound(min, max):
    return (min - BOUND_NOISE, max + BOUND_NOISE)

# Days between infections
# We infer the days between infections based on effective reproduction number (Rt)
def get_days_between_infections_in_begin(
        date_begin,
        date_end,
        days_to_recovery,
        df, date_column,
        effective_reproduction_number_column):
    df_outbreak = df[(df[date_column] >= date_begin) & (df[date_column] < date_end)]

    R_max = df_outbreak[effective_reproduction_number_column].max()
    R_mean = df_outbreak[effective_reproduction_number_column].mean()

    days_between_infections_0_min = days_to_recovery / R_max
    days_between_infections_0_max = days_to_recovery / R_mean
    days_between_infections_default = (days_between_infections_0_min + days_between_infections_0_max) / 2

    return days_between_infections_0_min, days_between_infections_default, days_between_infections_0_max

def get_max_days_between_infections_in_outbreak_adjustment(date_begin,
                                                           date_end,
                                                           days_to_recovery,
                                                           df,
                                                           date_column,
                                                           effective_reproduction_number_column):
    df_period = df[(df[date_column] >= date_begin) & (df[date_column] < date_end)]
    if len(df_period) == 0:
        df_period = df[(df[date_column] == date_begin)]

    R_min = df_period[effective_reproduction_number_column].quantile(0.25)

    days_between_infections_max = days_to_recovery / R_min

    return days_between_infections_max

def get_days_between_infections_after_first_outbreak(date_begin,
                                                     date_end,
                                                     days_to_recovery,
                                                     df,
                                                     days_between_infections_max_in_previous_outbreak,
                                                     date_column,
                                                     effective_reproduction_number_column):
    df_outbreak = df[(df[date_column] >= date_begin) & (df[date_column] < date_end)]

    R_max = df_outbreak[effective_reproduction_number_column].max()

    days_between_infections_min = days_to_recovery / (R_max / 0.3)
    days_between_infections_max = days_to_recovery / R_max

    days_between_infections_max_selected = min(days_between_infections_max,
                                               days_between_infections_max_in_previous_outbreak)

    if days_between_infections_max_selected < days_between_infections_min:
        temp = days_between_infections_max_selected
        days_between_infections_max_selected = days_between_infections_min
        days_between_infections_min = temp

    days_between_infections_default = (days_between_infections_min + days_between_infections_max_selected) / 2

    return days_between_infections_min, days_between_infections_default, days_between_infections_max_selected

# Case fatality probabilities
# For maximum bound, we use the ratio between deaths and cases in the period.
# For minimum bound we consider an extreme limit like whether all Brazilians had been infected by Covid-19 during the period.
# When case_fatality_max is an invalid value, we set the value 5 that is the max for national data
def get_case_fatality_probability(date_begin,
                                  date_end,
                                  df,
                                  date_column,
                                  effective_reproduction_number_column,
                                  death_column,
                                  location_population,
                                  case_column):
    df_period = df[(df[date_column] >= date_begin) & (df[date_column] < date_end)]

    deaths = df_period[death_column].sum()
    if deaths == 0:
        deaths = 1 / location_population * 100000
    cases = df_period[
        df_period[case_column] > 0][case_column].sum()
    if cases == 0:
        cases = 1 / location_population * 100000

    case_fatality_min = deaths / 100000  # 100,000 is the population because the deaths are in rate per 100,000 inhabitants infected
    if case_fatality_min < 0.0001:
        case_fatality_min = 0.0001
    case_fatality_max = deaths / cases
    if (case_fatality_min > case_fatality_max) | (case_fatality_max > 0.0133):
        case_fatality_max = 0.0133  # Verity, Robert, et al. "Estimates of the severity of coronavirus disease 2019: a model-based analysis." The Lancet infectious diseases 20.6 (2020): 669-677.
    case_fatality_default = (case_fatality_min + case_fatality_max) / 2

    return case_fatality_min, case_fatality_default, case_fatality_max

# This code cosiders case_column and death_column as rate by 100,000 inhabitants
def get_bounds_and_arguments(df,
                             date_column,
                             death_column,
                             effective_reproduction_number_column,
                             case_column,
                             outbreak_column,
                             days_to_recovery,
                             date_first_case,
                             max_date_to_fit,
                             location_population,
                             period_in_days,
                             cumulative_days_in_first_outbreak_to_max_bound_I0=14):
    # Breakpoints
    df_previous_outbreaks = df[df[date_column] <= max_date_to_fit].groupby([outbreak_column]).agg({date_column: ['min', 'max']})
    previous_outbreak_begins = df_previous_outbreaks[(date_column, 'min')].values
    previous_outbreak_begins = np.array([pd.to_datetime(breakpoint) for breakpoint in previous_outbreak_begins])
    previous_outbreak_ends = df_previous_outbreaks[(date_column, 'max')].values
    previous_outbreak_ends = np.array([pd.to_datetime(breakpoint) for breakpoint in previous_outbreak_ends])

    first_outbreak_begin = previous_outbreak_begins[0]

    min_breakpoints = previous_outbreak_begins
    max_breakpoints_based_peak = []

    for i in range(0,len(previous_outbreak_begins)):
        max_rt_in_breakpoint = df[(df[date_column] >= previous_outbreak_begins[i]) & (df[date_column] < previous_outbreak_ends[i])][effective_reproduction_number_column].max()
        date_max_rt_in_breakpoint = df[df[effective_reproduction_number_column] == max_rt_in_breakpoint][date_column].max()
        max_breakpoints_based_peak = np.append(max_breakpoints_based_peak, date_max_rt_in_breakpoint)

    # Bounds and default values
    # Number of initial infections
    min_initial_infected = 1 / location_population * 100000
    if cumulative_days_in_first_outbreak_to_max_bound_I0 != None:
        max_initial_infected = df[(df[date_column] >= date_first_case) & (
                df[date_column] <= first_outbreak_begin + timedelta(days=cumulative_days_in_first_outbreak_to_max_bound_I0))][case_column].sum()
    else:
        max_initial_infected = 1 / location_population * 100000
    default_initial_infected = min_initial_infected + (max_initial_infected - min_initial_infected) / 2
    bounds_initial_infected_population = create_bound(min_initial_infected, max_initial_infected)

    bounds_loss_immunity_in_days = create_bound(MIN_LOSS_IMMUNITY_IN_DAYS, MAX_LOSS_IMMUNITY_IN_DAYS)
    bounds_transition_days_between_epidemic_periods = create_bound(MIN_TRANSITION_DAYS_BETWEEN_EPIDEIC_PERIODS, MAX_TRANSITION_DAYS_BETWEEN_EPIDEIC_PERIODS)

    # Initial days between infections
    min_days_between_infections_0, default_days_between_infections_0, max_days_between_infections_0 = (
        get_days_between_infections_in_begin(previous_outbreak_begins[0],
                                             previous_outbreak_ends[0],
                                             days_to_recovery,
                                             df,
                                             date_column,
                                             effective_reproduction_number_column))
    bounds_days_between_infections_0 = create_bound(min_days_between_infections_0, max_days_between_infections_0)

    bounds = [bounds_initial_infected_population,
              bounds_days_between_infections_0]

    list_need_adjust_outbreak = np.array([])
    list_bounds_days_between_infection = []

    last_epidemic_period_with_slow_transition_begin = date_first_case
    list_bounds_case_fatality_probability = []
    list_bounds_loss_immunity_in_days = []
    list_breakpoints_in_slow_transition = []

    # Previous outbreaks
    for i in range(len(previous_outbreak_begins)):
        outbreak_begin = previous_outbreak_begins[i]
        outbreak_end = previous_outbreak_ends[i]

        if i + 1 < len(previous_outbreak_begins):
            next_outbreak_begin = previous_outbreak_begins[i + 1]
        else:
            next_outbreak_begin = max_date_to_fit

        rt_in_outbreak = df[
            (df[date_column] >= outbreak_begin) &
            (df[date_column] <= outbreak_end)][effective_reproduction_number_column].values

        if i == 0:
            need_adjust = need_adjust_first_outbreak(rt_in_outbreak, date_first_case, outbreak_end,
                                                     default_initial_infected, default_days_between_infections_0,
                                                     days_to_recovery, REFERENCE_CASE_FATALITY_RATE,
                                                     DEFAULT_LOSS_IMMUNITY_IN_DAYS)
            list_need_adjust_outbreak = np.append(list_need_adjust_outbreak, need_adjust)
            if need_adjust:
                max_days_between_infections = get_max_days_between_infections_in_outbreak_adjustment(
                    outbreak_end,
                    next_outbreak_begin,
                    days_to_recovery,
                    df,
                    date_column,
                    effective_reproduction_number_column)
                bounds_days_between_infections = create_bound(min_days_between_infections_0, max_days_between_infections)
                list_bounds_days_between_infection.append(bounds_days_between_infections)
        else:
            if len(list_bounds_days_between_infection) > 0:
                days_between_infections_max_in_previous_outbreak = list_bounds_days_between_infection[-1][1]
            else:
                days_between_infections_max_in_previous_outbreak = max_days_between_infections_0
            min_days_between_infections, default_days_between_infections, max_days_between_infections = (
                get_days_between_infections_after_first_outbreak(outbreak_begin,
                                                                 outbreak_end,
                                                                 days_to_recovery,
                                                                 df,
                                                                 days_between_infections_max_in_previous_outbreak,
                                                                 date_column,
                                                                 effective_reproduction_number_column))
            bounds_days_between_infections = create_bound(min_days_between_infections, max_days_between_infections)
            list_bounds_days_between_infection.append(bounds_days_between_infections)

            if i < (len(previous_outbreak_begins) - 1):
                need_adjust = need_adjust_secondary_outbreak(rt_in_outbreak)
            elif df[df[date_column] == max_date_to_fit][outbreak_column].isnull().values[0] == True:  # Check if there is occurring an outbreak at last day to fit
                need_adjust = need_adjust_secondary_outbreak(rt_in_outbreak)
            else:
                need_adjust = False  # When is occurring an outbreak at last day to fit, we not adjust the breakpoint

            list_need_adjust_outbreak = np.append(list_need_adjust_outbreak, need_adjust)
            if need_adjust:
                max_days_between_infections = get_max_days_between_infections_in_outbreak_adjustment(outbreak_end,
                                                                                                     next_outbreak_begin,
                                                                                                     days_to_recovery,
                                                                                                     df,
                                                                                                     date_column,
                                                                                                     effective_reproduction_number_column)
                bounds_days_between_infections = create_bound(min_days_between_infections, max_days_between_infections)
                list_bounds_days_between_infection.append(bounds_days_between_infections)

        interval = (outbreak_begin - last_epidemic_period_with_slow_transition_begin).days
        if interval >= MIN_INTERVAL_BETWEEN_EPIDEMIC_PERIOD_WITH_SLOW_TRANSITION:
            case_fatality_min, case_fatality_default, case_fatality_max = get_case_fatality_probability(
                last_epidemic_period_with_slow_transition_begin,
                outbreak_begin,
                df,
                date_column,
                effective_reproduction_number_column,
                death_column,
                location_population,
                case_column)
            bounds_case_fatality_probability = create_bound(case_fatality_min, case_fatality_max)
            list_bounds_case_fatality_probability.append(bounds_case_fatality_probability)
            list_bounds_loss_immunity_in_days.append(bounds_loss_immunity_in_days)
            if i == 0:
                list_breakpoints_in_slow_transition.append(0)
            elif need_adjust:
                list_breakpoints_in_slow_transition.append(len(list_bounds_days_between_infection) - 2)
            else:
                list_breakpoints_in_slow_transition.append(len(list_bounds_days_between_infection) - 1)
            last_epidemic_period_with_slow_transition_begin = outbreak_begin

    case_fatality_min, case_fatality_default, case_fatality_max = get_case_fatality_probability(
        last_epidemic_period_with_slow_transition_begin,
        max_date_to_fit,
        df,
        date_column,
        effective_reproduction_number_column,
        death_column,
        location_population,
        case_column)
    bounds_case_fatality_probability = create_bound(case_fatality_min, case_fatality_max)
    list_bounds_case_fatality_probability.append(bounds_case_fatality_probability)
    list_bounds_loss_immunity_in_days.append(bounds_loss_immunity_in_days)
    last_epidemic_period_with_slow_transition_begin = max_date_to_fit

    for bound in list_bounds_case_fatality_probability:
        bounds.append(bound)

    for bound in list_bounds_loss_immunity_in_days:
        bounds.append(bound)

    for bound in list_bounds_days_between_infection:
        bounds.append(bound)

    for i in range(len(min_breakpoints)):
        if (i == 0) & (list_need_adjust_outbreak[i] == True):
            bounds_breakpoint = create_bound(
                max((min_breakpoints[0]-date_first_case).days,0),
                (previous_outbreak_ends[0] - date_first_case).days)
            bounds.append(bounds_breakpoint)
        else:
            bounds_breakpoint_outbreak = create_bound(max((min_breakpoints[i]-date_first_case).days,0),
                                                      (max_breakpoints_based_peak[i] - date_first_case).days)
            bounds.append(bounds_breakpoint_outbreak)
            if list_need_adjust_outbreak[i]:
                bounds_breakpoint_outbreak_adjustment = create_bound(
                    (max_breakpoints_based_peak[i] - date_first_case).days,
                    (previous_outbreak_ends[i] - date_first_case).days)
                bounds.append(bounds_breakpoint_outbreak_adjustment)

    for i in range(len(min_breakpoints)):
        if (i == 0) & (list_need_adjust_outbreak[i]==True):
            bounds.append(bounds_transition_days_between_epidemic_periods)
        else:
            bounds.append(bounds_transition_days_between_epidemic_periods)
            if list_need_adjust_outbreak[i]:
                bounds.append(bounds_transition_days_between_epidemic_periods)

    real_new_deaths = df[(df[date_column] >= date_first_case) &
                         (df[date_column] <= max_date_to_fit)][death_column].round(6).values
    case_evolution_data = df[(df[date_column] >= date_first_case) &
                             (df[date_column] <= max_date_to_fit)][effective_reproduction_number_column].round(6).values

    quantity_outbreaks = len(previous_outbreak_begins)
    quantity_outbreak_adjustments = len(list_need_adjust_outbreak[list_need_adjust_outbreak == True])

    args = [period_in_days,
            real_new_deaths,
            case_evolution_data,
            days_to_recovery,
            list_breakpoints_in_slow_transition,
            quantity_outbreaks,
            quantity_outbreak_adjustments]

    return bounds, args


def get_error_deaths_rt(SIRDS_new_deaths,
                        real_new_deaths,
                        SIRDS_effective_reproduction_number,
                        real_effective_reproduction_number):
    try:
        mae_deaths = mean_absolute_error(real_new_deaths, SIRDS_new_deaths[:len(real_new_deaths)])
    except:
        try:
            mae_deaths = mean_absolute_error(real_new_deaths[1:], SIRDS_new_deaths[:len(real_new_deaths[1:])])
        except ValueError:
            if len(SIRDS_new_deaths) < len(real_new_deaths[1:]):
                resized_SIRDS_new_deaths = np.empty(len(real_new_deaths[1:]))
                resized_SIRDS_new_deaths[:] = np.nan
                resized_SIRDS_new_deaths[:len(SIRDS_new_deaths)] = SIRDS_new_deaths
                resized_SIRDS_new_deaths = np.nan_to_num(resized_SIRDS_new_deaths, nan=0)
                mae_deaths = mean_absolute_error(real_new_deaths[1:], resized_SIRDS_new_deaths)

    indices_to_remove_rt = np.argwhere(np.isnan(real_effective_reproduction_number))
    real_effective_reproduction_number = np.delete(real_effective_reproduction_number, indices_to_remove_rt)
    SIRDS_effective_reproduction_number = np.delete(SIRDS_effective_reproduction_number, indices_to_remove_rt)
    try:
        mae_rt = mean_absolute_error(real_effective_reproduction_number,
                                     SIRDS_effective_reproduction_number[:len(real_effective_reproduction_number)])
    except ValueError:
        if len(SIRDS_effective_reproduction_number) < len(real_effective_reproduction_number):
            resized_SIRDS_effective_reproduction_number = np.empty(len(real_effective_reproduction_number))
            resized_SIRDS_effective_reproduction_number[:] = np.nan
            resized_SIRDS_effective_reproduction_number[
            :len(SIRDS_effective_reproduction_number)] = SIRDS_effective_reproduction_number
            resized_SIRDS_effective_reproduction_number = np.nan_to_num(resized_SIRDS_effective_reproduction_number,
                                                                        nan=0)
            mae_rt = mean_absolute_error(real_effective_reproduction_number,
                                         resized_SIRDS_effective_reproduction_number)

    error = mae_deaths / real_new_deaths[1:].mean() + mae_rt / real_effective_reproduction_number.mean()

    return error


def sirds_objective_function(x, *args):
    period_in_days = args[0]
    real_new_deaths = args[1]
    case_evolution_data = args[2]
    days_to_recovery = args[3]
    list_breakpoints_in_slow_transition = args[4]

    initial_infected_population = x[0]

    # Initial SIRDS parameters
    days_between_infections = x[1]
    requirement_values = [initial_infected_population, period_in_days, days_between_infections, days_to_recovery,
                          list_breakpoints_in_slow_transition]

    quantity_epidemic_periods_with_slow_transition = len(list_breakpoints_in_slow_transition) + 1

    case_fatality_probability_values = x[2:2 + quantity_epidemic_periods_with_slow_transition]
    loss_immunity_in_days_values = x[
                                   2 + quantity_epidemic_periods_with_slow_transition:2 + 2 * quantity_epidemic_periods_with_slow_transition]

    x_breakpoint_params = x[2 + 2 * quantity_epidemic_periods_with_slow_transition:]
    num_breakpoints = int(len(x_breakpoint_params) / 3)

    if num_breakpoints > 0:
        # Extract parameters for each breakpoint
        days_between_infections_values = x_breakpoint_params[:num_breakpoints]
        breakpoint_values = x_breakpoint_params[num_breakpoints:2 * num_breakpoints]
        transition_days_between_epidemic_periods_values = x_breakpoint_params[2 * num_breakpoints:]
    else:
        days_between_infections_values = []
        breakpoint_values = []
        transition_days_between_epidemic_periods_values = []

    y = sirds_model_fuzzy(tuple([*requirement_values, *case_fatality_probability_values, *loss_immunity_in_days_values,
                                 *days_between_infections_values, *breakpoint_values,
                                 *transition_days_between_epidemic_periods_values]))

    S, I, R, D, I_accumulated = y
    N = S[0] + I[0] + R[0] + D[0]

    SIRDS_new_deaths = get_new_deaths(D)

    days_between_infections_values_full = np.append([days_between_infections], days_between_infections_values)

    fuzzy_transition = get_epidemic_periods_for_beta_transition_fuzzy_variable(period_in_days, breakpoint_values,
                                                                               transition_days_between_epidemic_periods_values)

    SIRDS_effective_reproduction_number = get_fuzzy_effective_reproduction_number(S, N, tuple(
        [*[days_to_recovery, (fuzzy_transition, days_between_infections_values_full)]]))

    return get_error_deaths_rt(SIRDS_new_deaths, real_new_deaths, SIRDS_effective_reproduction_number,
                               case_evolution_data)