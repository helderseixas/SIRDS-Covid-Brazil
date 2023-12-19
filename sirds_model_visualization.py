import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from PIL import Image
from skfuzzy.control.visualization import FuzzyVariableVisualizer
import skfuzzy as fuzz
import util
from sirds_model import sirds_model_fuzzy, get_new_deaths, get_new_cases, \
    get_epidemic_periods_for_beta_transition_fuzzy_variable, get_epidemic_periods_with_slow_transition_fuzzy_variable, \
    get_fuzzy_effective_reproduction_number, epidemic_parameter_defuzzification, get_error_deaths_rt


def get_sirds(result):
    x_days_between_infections = result.filter(like='x_days_between_infections').values.tolist()[1:]
    x_case_fatality_probability = result.filter(like='x_case_fatality_probability').values.tolist()
    x_loss_immunity_in_days = result.filter(like='x_loss_immunity_in_days').values.tolist()
    x_breakpoint = result.filter(like='x_breakpoint').values.tolist()
    x_transition_days_between_epidemic_periods = result.filter(
        like='x_transition_days_between_epidemic_periods').values.tolist()
    list_breakpoints_in_slow_transition = ast.literal_eval(result.list_breakpoints_in_slow_transition)

    y = sirds_model_fuzzy(tuple([*[
        result.x_initial_infected_population,
        result.period_in_days,
        result.x_days_between_infections_0,
        result.days_to_recovery, list_breakpoints_in_slow_transition], *x_case_fatality_probability,
                                 *x_loss_immunity_in_days, *x_days_between_infections, *x_breakpoint,
                                 *x_transition_days_between_epidemic_periods]))

    return y


def get_sirds_extras(result, S, D, I_accumulated):
    period_in_days = int(result.period_in_days)
    D_new_deaths = get_new_deaths(D)
    I_new_cases = get_new_cases(I_accumulated)

    x_days_between_infections = result.filter(like='x_days_between_infections').values.tolist()[1:]
    x_breakpoint = result.filter(like='x_breakpoint').values.tolist()
    x_case_fatality_probability = result.filter(like='x_case_fatality_probability').values.tolist()
    x_loss_immunity_in_days = result.filter(like='x_loss_immunity_in_days').values.tolist()
    x_transition_days_between_epidemic_periods = result.filter(
        like='x_transition_days_between_epidemic_periods').values.tolist()

    days_between_infections_values_full = np.append([result.x_days_between_infections_0], x_days_between_infections)

    epidemic_periods_with_fast_transition_fuzzy_variable = get_epidemic_periods_for_beta_transition_fuzzy_variable(
        result.period_in_days, x_breakpoint, x_transition_days_between_epidemic_periods)

    list_breakpoints_in_slow_transition = ast.literal_eval(result.list_breakpoints_in_slow_transition)
    slow_transition_breakpoint_values = []
    for indice_breakpoint in list_breakpoints_in_slow_transition:
        slow_transition_breakpoint_values.append(x_breakpoint[indice_breakpoint])

    epidemic_periods_with_slow_transition_fuzzy_variable = get_epidemic_periods_with_slow_transition_fuzzy_variable(
        result.period_in_days, slow_transition_breakpoint_values)

    SIRDS_effective_reproduction_number = get_fuzzy_effective_reproduction_number(S, 100000, tuple([*[
        result.days_to_recovery,
        (epidemic_periods_with_fast_transition_fuzzy_variable, days_between_infections_values_full)]]))

    estimated_days_between_infections = np.array([])
    for i in range(period_in_days):
        estimation = epidemic_parameter_defuzzification(epidemic_periods_with_fast_transition_fuzzy_variable,
                                                  days_between_infections_values_full, i)
        estimated_days_between_infections = np.append(estimated_days_between_infections, [estimation])

    estimated_case_fatality_probability = np.array([])
    for i in range(period_in_days):
        estimation = epidemic_parameter_defuzzification(epidemic_periods_with_slow_transition_fuzzy_variable,
                                                  x_case_fatality_probability, i)
        estimated_case_fatality_probability = np.append(estimated_case_fatality_probability, [estimation])

    estimated_loss_immunity_in_days = np.array([])
    for i in range(period_in_days):
        estimation = epidemic_parameter_defuzzification(epidemic_periods_with_slow_transition_fuzzy_variable,
                                                  x_loss_immunity_in_days, i)
        estimated_loss_immunity_in_days = np.append(estimated_loss_immunity_in_days, [estimation])

    return (D_new_deaths,
            SIRDS_effective_reproduction_number,
            I_new_cases,
            epidemic_periods_with_fast_transition_fuzzy_variable,
            epidemic_periods_with_slow_transition_fuzzy_variable,
            days_between_infections_values_full,
            x_case_fatality_probability,
            x_loss_immunity_in_days,
            estimated_days_between_infections,
            estimated_case_fatality_probability,
            estimated_loss_immunity_in_days)

def _calculate_performance(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird):
    if len(real_reproduction_number[~np.isnan(real_reproduction_number)]) > 0:
        mae = get_error_deaths_rt(D_new_deaths,
                            real_new_deaths,
                            reproduction_number_sird,
                            real_reproduction_number)

        indices_to_remove = np.argwhere(np.isnan(real_reproduction_number))
        real_reproduction_number = np.delete(real_reproduction_number, indices_to_remove)
        reproduction_number_sird_train = np.delete(reproduction_number_sird, indices_to_remove)
        sse_Rt = mean_squared_error(real_reproduction_number, reproduction_number_sird_train)
        r2_Rt = r2_score(real_reproduction_number, reproduction_number_sird_train)
    else:
        mae = None
        sse_Rt = None
        r2_Rt = None

    sse_D = mean_squared_error(D_new_deaths, real_new_deaths)
    r2_D = r2_score(D_new_deaths, real_new_deaths)

    return mae, sse_D, r2_D, sse_Rt, r2_Rt

def _show_performance(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird):
    mae = get_error_deaths_rt(D_new_deaths,
                        real_new_deaths,
                        reproduction_number_sird,
                        real_reproduction_number)

    print('\nGeneral MAE: ' + str(round(mae, 3)))

    print('\nNew deaths:')
    sse = mean_squared_error(D_new_deaths, real_new_deaths)
    r2 = r2_score(D_new_deaths, real_new_deaths)
    print('SSE: ' + str(round(sse, 3)))
    print('r2: ' + str(round(r2, 3)))

    print('\nRt:')
    indices_to_remove = np.argwhere(np.isnan(real_reproduction_number))
    real_reproduction_number = np.delete(real_reproduction_number, indices_to_remove)
    reproduction_number_sird_train = np.delete(reproduction_number_sird, indices_to_remove)
    sse = mean_squared_error(real_reproduction_number, reproduction_number_sird_train)
    print('SSE: ' + str(round(sse, 3)))
    r2 = r2_score(real_reproduction_number, reproduction_number_sird_train)
    print('r2: ' + str(round(r2, 3)))

def calculate_performance(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird, train_period=None):
    if train_period is None:
        train_period = len(real_new_deaths)

    if len(D_new_deaths) < len(real_new_deaths):
        D_fitted = D_new_deaths[:train_period-1]
        real_new_deaths_fitted = real_new_deaths[1:train_period]
    else:
        D_fitted = D_new_deaths[1:train_period]
        real_new_deaths_fitted = real_new_deaths[1:train_period]

    reproduction_number_sird_fitted = reproduction_number_sird[:train_period]
    real_reproduction_number_fitted = real_reproduction_number[:train_period]
    mae_fit, sse_D_fit, r2_D_fit, sse_Rt_fit, r2_Rt_fit = _calculate_performance(
        real_new_deaths_fitted, D_fitted, real_reproduction_number_fitted, reproduction_number_sird_fitted)

    if train_period < len(real_new_deaths):
        if len(D_new_deaths) < len(real_new_deaths):
            D_predicted = D_new_deaths[train_period-1:]
        else:
            D_predicted = D_new_deaths[train_period:]
        real_new_deaths_predicted = real_new_deaths[train_period:]
        D_predicted = D_predicted[:len(real_new_deaths_predicted)]

        real_reproduction_number_predicted = real_reproduction_number[train_period:]
        reproduction_number_sird_predicted = reproduction_number_sird[train_period:]
        reproduction_number_sird_predicted = reproduction_number_sird_predicted[:len(real_reproduction_number_predicted)]

        mae_predicton, sse_D_predicton, r2_D_predicton, sse_Rt_predicton, r2_Rt_predicton = _calculate_performance(
            real_new_deaths_predicted, D_predicted, real_reproduction_number_predicted, reproduction_number_sird_predicted)

        real_new_deaths_predicted_month_1 = real_new_deaths_predicted[:30]
        D_predicted_month_1 = D_predicted[:30]
        real_reproduction_number_predicted_month_1 = real_reproduction_number_predicted[:30]
        reproduction_number_sird_predicted_month_1 = reproduction_number_sird_predicted[:30]
        mae_predicton_month_1, sse_D_predicton_month_1, r2_D_predicton_month_1, sse_Rt_predicton_month_1, r2_Rt_predicton_month_1 = _calculate_performance(
            real_new_deaths_predicted_month_1, D_predicted_month_1, real_reproduction_number_predicted_month_1, reproduction_number_sird_predicted_month_1)

        real_new_deaths_predicted_month_2 = real_new_deaths_predicted[30:60]
        D_predicted_month_2 = D_predicted[30:60]
        real_reproduction_number_predicted_month_2 = real_reproduction_number_predicted[30:60]
        reproduction_number_sird_predicted_month_2 = reproduction_number_sird_predicted[30:60]
        mae_predicton_month_2, sse_D_predicton_month_2, r2_D_predicton_month_2, sse_Rt_predicton_month_2, r2_Rt_predicton_month_2 = _calculate_performance(
            real_new_deaths_predicted_month_2, D_predicted_month_2, real_reproduction_number_predicted_month_2, reproduction_number_sird_predicted_month_2)

        real_new_deaths_predicted_month_3 = real_new_deaths_predicted[60:]
        D_predicted_month_3 = D_predicted[60:]
        real_reproduction_number_predicted_month_3 = real_reproduction_number_predicted[60:]
        reproduction_number_sird_predicted_month_3 = reproduction_number_sird_predicted[60:]
        mae_predicton_month_3, sse_D_predicton_month_3, r2_D_predicton_month_3, sse_Rt_predicton_month_3, r2_Rt_predicton_month_3 = (
                _calculate_performance(real_new_deaths_predicted_month_3, D_predicted_month_3, real_reproduction_number_predicted_month_3, reproduction_number_sird_predicted_month_3))

        return mae_fit, sse_D_fit, r2_D_fit, sse_Rt_fit, r2_Rt_fit, mae_predicton, sse_D_predicton, r2_D_predicton, sse_Rt_predicton, r2_Rt_predicton, mae_predicton_month_1, sse_D_predicton_month_1, r2_D_predicton_month_1, sse_Rt_predicton_month_1, r2_Rt_predicton_month_1, mae_predicton_month_2, sse_D_predicton_month_2, r2_D_predicton_month_2, sse_Rt_predicton_month_2, r2_Rt_predicton_month_2, mae_predicton_month_3, sse_D_predicton_month_3, r2_D_predicton_month_3, sse_Rt_predicton_month_3, r2_Rt_predicton_month_3
    else:
        return mae_fit, sse_D_fit, r2_D_fit, sse_Rt_fit, r2_Rt_fit

def show_performance(dict_performance):
    for measurement, values in dict_performance.items():
        mean = np.mean(values)
        lower_bound, upper_bound = util.calculate_confidence_interval(values)
        print(measurement,': ', mean, '(', lower_bound, ',', upper_bound, ')')

def plot_result(df_S, df_I, df_R, df_D, df_new_deaths, df_I_accumulated, real_new_deaths, real_total_deaths,
                real_reproduction_number, df_rt,
                real_total_cases, real_new_cases, df_new_cases, dates, directory_to_save='images',
                id_in_file='', max_date_to_fit=None):

    mask_date = mdates.DateFormatter('%m/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    plt.rc('font', size=8)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())
    fig, ax = plt.subplots(3, 2, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(14.288)), sharex=False)

    # Plot the data on three separate curves for S(t), I(t), R(t) and D(t)
    sns.lineplot(x=df_S['date'], y=df_S['S'], label='Susceptible', color=util.get_default_colors_categorical_seaborn()[1], legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_I['date'], y=df_I['I'], label='Infected', color=util.get_default_colors_categorical_seaborn()[2], legend=True,
                 linestyle=line_styles[1], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_R['date'], y=df_R['R'], label='Recovered', color=util.get_default_colors_categorical_seaborn()[4], legend=True,
                 linestyle=line_styles[2], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_D['date'], y=df_D['D'], label='Deceased', color=util.get_default_colors_categorical_seaborn()[3], legend=True,
                 linestyle=line_styles[3], ax=ax.flatten()[0], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[0].set_xlabel('Month/Year')
    ax.flatten()[0].xaxis.set_major_formatter(mask_date)
    ax.flatten()[0].tick_params(axis='x', labelrotation=20)
    ax.flatten()[0].set_ylabel('Population')
    ax.flatten()[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[0].set_title('a) SIRDS simulation')
    ax.flatten()[0].legend()

    # Plot Rt
    sns.lineplot(x=dates, y=real_reproduction_number, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[1])
    sns.lineplot(x=df_rt['date'], y=df_rt['rt'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[1], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[1].axhline(1, 0, 1, linestyle='--', color='red')
    ax.flatten()[1].set_xlabel('Month/Year')
    ax.flatten()[1].xaxis.set_major_formatter(mask_date)
    ax.flatten()[1].tick_params(axis='x', labelrotation=20)
    ax.flatten()[1].set_ylabel('$R_{t}$')
    ax.flatten()[1].set_title('b) Effective reproduction number ($R_{t}$)')
    ax.flatten()[1].legend()

    # Plot new cases
    sns.lineplot(x=dates, y=real_new_cases, label='Original data (reported cases)', legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[2])
    sns.lineplot(x=df_new_cases['date'], y=df_new_cases['cases'], label='Simulation (infections)', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[2], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[2].set_xlabel('Month/Year')
    ax.flatten()[2].xaxis.set_major_formatter(mask_date)
    ax.flatten()[2].tick_params(axis='x', labelrotation=20)
    ax.flatten()[2].set_ylabel('Events per 100,000 people')
    ax.flatten()[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[2].set_title('c) New cases (infections)')
    ax.flatten()[2].legend()

    # Plot new deaths
    sns.lineplot(x=dates, y=real_new_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[3])
    sns.lineplot(x=df_new_deaths['date'], y=df_new_deaths['deaths'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[3], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[3].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[3].set_xlabel('Month/Year')
    ax.flatten()[3].xaxis.set_major_formatter(mask_date)
    ax.flatten()[3].tick_params(axis='x', labelrotation=20)
    ax.flatten()[3].set_ylabel('Deaths per 100,000 people')
    ax.flatten()[3].set_title('d) New deaths')
    ax.flatten()[3].legend()

    # Plot total cases
    sns.lineplot(x=dates, y=real_total_cases, label='Original data (reported cases)', legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[4])
    sns.lineplot(x=df_I_accumulated['date'], y=df_I_accumulated['I_accumulated'], label='Simulation (infections)', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[4], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[4].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[4].set_xlabel('Month/Year')
    ax.flatten()[4].xaxis.set_major_formatter(mask_date)
    ax.flatten()[4].tick_params(axis='x', labelrotation=20)
    ax.flatten()[4].set_ylabel('Events per 100,000 people')
    ax.flatten()[4].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[4].set_title('e) Total cases (infections)')
    ax.flatten()[4].legend()

    # Plot total deaths
    sns.lineplot(x=dates, y=real_total_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[5])
    sns.lineplot(x=df_D['date'], y=df_D['D'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[5], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[5].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[5].set_xlabel('Month/Year')
    ax.flatten()[5].xaxis.set_major_formatter(mask_date)
    ax.flatten()[5].tick_params(axis='x', labelrotation=20)
    ax.flatten()[5].set_ylabel('Deaths per 100,000 people')
    ax.flatten()[5].set_title('f) Total deaths')
    ax.flatten()[5].legend()

    fig.tight_layout()
    filename = 'images/result_output'+id_in_file
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def plot_outbreak_0_result(dict_outbreak_S,
                         dict_outbreak_I,
                         dict_outbreak_R,
                         dict_outbreak_D,
                         dict_outbreak_rt,
                         dict_outbreak_new_deaths,
                         dict_max_date_to_fit,
                         real_reproduction_number,
                         real_new_deaths):
    mask_date = mdates.DateFormatter('%m/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    plt.rc('font', size=8)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())

    fig, ax = plt.subplots(8, 3, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(22.23)), sharex=True)

    list_days_to_fit = dict_outbreak_S.keys()
    for j, days_to_fit in enumerate(list_days_to_fit):
        days_to_fit = int(days_to_fit)
        df_S = dict_outbreak_S[days_to_fit]
        df_I = dict_outbreak_I[days_to_fit]
        df_R = dict_outbreak_R[days_to_fit]
        df_D = dict_outbreak_D[days_to_fit]
        max_date_to_fit = dict_max_date_to_fit[days_to_fit]

        dates = df_S.date.unique()
        if len(real_reproduction_number) < len(dates):
            dates = dates[:len(real_reproduction_number)]

        # Chart 0: SIRDS output
        sns.lineplot(x=df_S['date'], y=df_S['S'], label='S',
                     color=util.get_default_colors_categorical_seaborn()[1], legend=days_to_fit==0,
                     linestyle=line_styles[0], ax=ax[j][0], errorbar=('ci', 95))
        sns.lineplot(x=df_I['date'], y=df_I['I'], label='I',
                     color=util.get_default_colors_categorical_seaborn()[2], legend=days_to_fit==0,
                     linestyle=line_styles[1], ax=ax[j][0], errorbar=('ci', 95))
        sns.lineplot(x=df_R['date'], y=df_R['R'], label='R',
                     color=util.get_default_colors_categorical_seaborn()[4], legend=days_to_fit==0,
                     linestyle=line_styles[2], ax=ax[j][0], errorbar=('ci', 95))
        sns.lineplot(x=df_D['date'], y=df_D['D'], label='D',
                     color=util.get_default_colors_categorical_seaborn()[3], legend=days_to_fit==0,
                     linestyle=line_styles[3], ax=ax[j][0], errorbar=('ci', 95))
        ax[j][0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        if str(days_to_fit)[-1] == '1':
            ordinal_symbol = '$^{st}$'
        elif str(days_to_fit)[-1] == '2':
            ordinal_symbol = '$^{nd}$'
        elif str(days_to_fit)[-1] == '3':
            ordinal_symbol = '$^{rd}$'
        else:
            ordinal_symbol = '$^{th}$'
        ax[j][0].set_ylabel(str(days_to_fit)+ordinal_symbol+' day\nPopulation')
        ax[j][0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        if j == 0:
            ax[j][0].set_title('a) SIRDS Simulation')
            ax[j][0].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.3, 1.75))

        # Chart 1: Rt
        df_rt = dict_outbreak_rt[days_to_fit]
        sns.lineplot(x=dates, y=real_reproduction_number[:len(dates)], label='Original data', legend=days_to_fit==0, linestyle=line_styles[0],
                     ax=ax[j][1])
        sns.lineplot(x=df_rt['date'], y=df_rt['rt'], label='Simulation', legend=days_to_fit==0, linestyle=line_styles[1],
                     ax=ax[j][1], errorbar=('ci', 95))
        ax[j][1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[j][1].axhline(1, 0, 1, linestyle='--', color='red')
        ax[j][1].set_ylabel('$R_{t}$')
        if j == 0:
            ax[j][1].set_title('b) Effective reproduction number ($R_t$)')
            ax[j][1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.75))

        # Chart 2: Deaths
        df_new_deaths = dict_outbreak_new_deaths[days_to_fit]
        sns.lineplot(x=dates, y=real_new_deaths[:len(dates)], label='Original data', legend=days_to_fit==0, linestyle=line_styles[0],
                     ax=ax[j][2])
        sns.lineplot(x=df_new_deaths['date'], y=df_new_deaths['deaths'], label='Simulation', legend=days_to_fit==0,
                     linestyle=line_styles[1],
                     ax=ax[j][2], errorbar=('ci', 95))
        ax[j][2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[j][2].set_ylabel('Death rate')
        if j == 0:
            ax[j][2].set_title('c) New deaths')
            ax[j][2].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.75))

        # General
        if j == len(list_days_to_fit) -1:
            for i in range(3):
                ax[j][i].set_xlabel('Month/Year')
                ax[j][i].xaxis.set_major_formatter(mask_date)
                ax[j][i].tick_params(axis='x', labelrotation=90)

    fig.tight_layout()
    filename = 'images/outbreak_0_result_output'
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def plot_outbreak_result(dict_outbreak_S,
                         dict_outbreak_I,
                         dict_outbreak_R,
                         dict_outbreak_D,
                         dict_outbreak_rt,
                         dict_outbreak_new_deaths,
                         dict_max_date_to_fit,
                         real_reproduction_number,
                         real_new_deaths):
    mask_date = mdates.DateFormatter('%m/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    plt.rc('font', size=8)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())

    fig, ax = plt.subplots(9, 3, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(22.23)), sharex=True)

    outbreaks = dict_outbreak_S.keys()
    for outbreak in outbreaks:
        outbreak = int(outbreak)
        df_S = dict_outbreak_S[outbreak]
        df_I = dict_outbreak_I[outbreak]
        df_R = dict_outbreak_R[outbreak]
        df_D = dict_outbreak_D[outbreak]
        max_date_to_fit = dict_max_date_to_fit[outbreak]

        dates = df_S.date.unique()
        if len(real_reproduction_number) < len(dates):
            dates = dates[:len(real_reproduction_number)]

        # Chart 0: SIRDS output
        sns.lineplot(x=df_S['date'], y=df_S['S'], label='S',
                     color=util.get_default_colors_categorical_seaborn()[1], legend=outbreak==0,
                     linestyle=line_styles[0], ax=ax[outbreak][0], errorbar=('ci', 95))
        sns.lineplot(x=df_I['date'], y=df_I['I'], label='I',
                     color=util.get_default_colors_categorical_seaborn()[2], legend=outbreak==0,
                     linestyle=line_styles[1], ax=ax[outbreak][0], errorbar=('ci', 95))
        sns.lineplot(x=df_R['date'], y=df_R['R'], label='R',
                     color=util.get_default_colors_categorical_seaborn()[4], legend=outbreak==0,
                     linestyle=line_styles[2], ax=ax[outbreak][0], errorbar=('ci', 95))
        sns.lineplot(x=df_D['date'], y=df_D['D'], label='D',
                     color=util.get_default_colors_categorical_seaborn()[3], legend=outbreak==0,
                     linestyle=line_styles[3], ax=ax[outbreak][0], errorbar=('ci', 95))
        ax[outbreak][0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[outbreak][0].set_ylabel('Outbreak '+str(outbreak)+'\nPopulation')
        ax[outbreak][0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        if outbreak == 0:
            ax[outbreak][0].set_title('a) SIRDS Simulation')
            ax[outbreak][0].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.3, 1.75))

        # Chart 1: Rt
        df_rt = dict_outbreak_rt[outbreak]
        sns.lineplot(x=dates, y=real_reproduction_number[:len(dates)], label='Original data', legend=outbreak==0, linestyle=line_styles[0],
                     ax=ax[outbreak][1])
        sns.lineplot(x=df_rt['date'], y=df_rt['rt'], label='Simulation', legend=outbreak==0, linestyle=line_styles[1],
                     ax=ax[outbreak][1], errorbar=('ci', 95))
        ax[outbreak][1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[outbreak][1].axhline(1, 0, 1, linestyle='--', color='red')
        ax[outbreak][1].set_ylabel('$R_{t}$')
        if outbreak == 0:
            ax[outbreak][1].set_title('b) Effective reproduction number ($R_t$)')
            ax[outbreak][1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.75))

        # Chart 2: Deaths
        df_new_deaths = dict_outbreak_new_deaths[outbreak]
        sns.lineplot(x=dates, y=real_new_deaths[:len(dates)], label='Original data', legend=outbreak==0, linestyle=line_styles[0],
                     ax=ax[outbreak][2])
        sns.lineplot(x=df_new_deaths['date'], y=df_new_deaths['deaths'], label='Simulation', legend=outbreak==0,
                     linestyle=line_styles[1],
                     ax=ax[outbreak][2], errorbar=('ci', 95))
        ax[outbreak][2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[outbreak][2].set_ylabel('Death rate')
        if outbreak == 0:
            ax[outbreak][2].set_title('c) New deaths')
            ax[outbreak][2].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.75))

        # General
        if outbreak == len(outbreaks) -1:
            for i in range(3):
                ax[outbreak][i].set_xlabel('Month/Year')
                ax[outbreak][i].xaxis.set_major_formatter(mask_date)
                ax[outbreak][i].tick_params(axis='x', labelrotation=90)

    fig.tight_layout()
    filename = 'images/outbreak_result_output'
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def plot_outbreak_result_forecasting_focus(dict_outbreak_S,
                         dict_outbreak_I,
                         dict_outbreak_R,
                         dict_outbreak_D,
                         dict_outbreak_rt,
                         dict_outbreak_new_deaths,
                         dict_max_date_to_fit,
                         real_reproduction_number,
                         real_new_deaths):
    mask_date = mdates.DateFormatter('%m/%d/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    plt.rc('font', size=8)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())

    fig, ax = plt.subplots(9, 3, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(22.23)), sharex=False)

    outbreaks = dict_outbreak_S.keys()
    for outbreak in outbreaks:
        outbreak = int(outbreak)
        df_S = dict_outbreak_S[outbreak]
        df_I = dict_outbreak_I[outbreak]
        df_R = dict_outbreak_R[outbreak]
        df_D = dict_outbreak_D[outbreak]
        max_date_to_fit = dict_max_date_to_fit[outbreak]
        df_rt = dict_outbreak_rt[outbreak]
        df_new_deaths = dict_outbreak_new_deaths[outbreak]

        min_date = df_S.date.min()
        outbreak_begin_date = max_date_to_fit - pd.DateOffset(days=20)
        days_before_outbreak_begin = (outbreak_begin_date - min_date).days
        forecasting_analysis_end = max_date_to_fit + pd.DateOffset(days=90)

        df_S = df_S[(df_S.date >= outbreak_begin_date) & (df_S.date <= forecasting_analysis_end)]
        df_I = df_I[(df_I.date >= outbreak_begin_date) & (df_I.date <= forecasting_analysis_end)]
        df_R = df_R[(df_R.date >= outbreak_begin_date) & (df_R.date <= forecasting_analysis_end)]
        df_D = df_D[(df_D.date >= outbreak_begin_date) & (df_D.date <= forecasting_analysis_end)]
        df_rt = df_rt[(df_rt.date >= outbreak_begin_date) & (df_rt.date <= forecasting_analysis_end)]
        df_new_deaths = df_new_deaths[(df_new_deaths.date >= outbreak_begin_date) &
                                      (df_new_deaths.date <= forecasting_analysis_end)]

        dates = df_S.date.unique()
        real_reproduction_number_outreak = real_reproduction_number[days_before_outbreak_begin:days_before_outbreak_begin+21+90]
        real_new_deaths_outreak = real_new_deaths[days_before_outbreak_begin:days_before_outbreak_begin + 21 + 90]
        if len(real_reproduction_number_outreak) < len(dates):
            dates = dates[:len(real_reproduction_number_outreak)]

        # Chart 0: SIRDS output
        sns.lineplot(x=df_S['date'], y=df_S['S'], label='S',
                     color=util.get_default_colors_categorical_seaborn()[1], legend=outbreak==0,
                     linestyle=line_styles[0], ax=ax[outbreak][0], errorbar=('ci', 95))
        sns.lineplot(x=df_I['date'], y=df_I['I'], label='I',
                     color=util.get_default_colors_categorical_seaborn()[2], legend=outbreak==0,
                     linestyle=line_styles[1], ax=ax[outbreak][0], errorbar=('ci', 95))
        sns.lineplot(x=df_R['date'], y=df_R['R'], label='R',
                     color=util.get_default_colors_categorical_seaborn()[4], legend=outbreak==0,
                     linestyle=line_styles[2], ax=ax[outbreak][0], errorbar=('ci', 95))
        sns.lineplot(x=df_D['date'], y=df_D['D'], label='D',
                     color=util.get_default_colors_categorical_seaborn()[3], legend=outbreak==0,
                     linestyle=line_styles[3], ax=ax[outbreak][0], errorbar=('ci', 95))
        ax[outbreak][0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[outbreak][0].set_ylim([0, 100000])
        ax[outbreak][0].set_ylabel('Outbreak '+str(outbreak)+'\nPopulation')
        ax[outbreak][0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax[outbreak][0].set_xlabel(None)
        ax[outbreak][0].xaxis.set_major_formatter(mask_date)
        ax[outbreak][0].set_xticks([max_date_to_fit])
        ax[outbreak][0].tick_params(axis='x', labelrotation=0)

        if outbreak == 0:
            ax[outbreak][0].set_title('a) SIRDS Simulation')
            ax[outbreak][0].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.3, 1.9))

        # Chart 1: Rt
        sns.lineplot(x=dates, y=real_reproduction_number_outreak, label='Original data', legend=outbreak==0, linestyle=line_styles[0],
                     ax=ax[outbreak][1])
        sns.lineplot(x=df_rt['date'], y=df_rt['rt'], label='Simulation', legend=outbreak==0, linestyle=line_styles[1],
                     ax=ax[outbreak][1], errorbar=('ci', 95))
        ax[outbreak][1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[outbreak][1].axhline(1, 0, 1, linestyle='--', color='red')
        ax[outbreak][1].set_ylabel('$R_{t}$')
        ax[outbreak][1].set_xlabel(None)
        ax[outbreak][1].xaxis.set_major_formatter(mask_date)
        ax[outbreak][1].set_xticks([max_date_to_fit])
        ax[outbreak][1].tick_params(axis='x', labelrotation=0)
        if outbreak == 0:
            ax[outbreak][1].set_title('b) Effective reproduction number ($R_t$)')
            ax[outbreak][1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.9))

        # Chart 2: Deaths
        sns.lineplot(x=dates, y=real_new_deaths_outreak, label='Original data', legend=outbreak==0, linestyle=line_styles[0],
                     ax=ax[outbreak][2])
        sns.lineplot(x=df_new_deaths['date'], y=df_new_deaths['deaths'], label='Simulation', legend=outbreak==0,
                     linestyle=line_styles[1],
                     ax=ax[outbreak][2], errorbar=('ci', 95))
        ax[outbreak][2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
        ax[outbreak][2].set_ylabel('Death rate')
        ax[outbreak][2].set_xlabel(None)
        ax[outbreak][2].xaxis.set_major_formatter(mask_date)
        ax[outbreak][2].set_xticks([max_date_to_fit])
        ax[outbreak][2].tick_params(axis='x', labelrotation=0)
        if outbreak == 0:
            ax[outbreak][2].set_title('c) New deaths')
            ax[outbreak][2].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.9))

        # General
        if outbreak == len(outbreaks) -1:
            for i in range(3):
                ax[outbreak][i].set_xlabel('Date')

    fig.tight_layout()
    filename = 'images/outbreak_result_output_forecasting_focus'
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def plot_fuzzy_variables(dates, list_fuzzy_fast_transition_variable, list_fuzzy_slow_transition_variable, country):
    myFmt = mdates.DateFormatter('%m/%Y')
    plt.rc('font', size=6)
    sns.set_style("ticks")

    fig, ax = plt.subplots(1, 2, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(4.76)))

    df_fuzzy_fast_transition_variable = pd.concat([
        pd.DataFrame({'date': dates, 'membership_values': term.mf, 'term_name': term_name})
        for fuzzy_variable in list_fuzzy_fast_transition_variable
        for term_name, term in fuzzy_variable.terms.items()
    ])

    df_fuzzy_slow_transition_variable = pd.concat([
        pd.DataFrame({'date': dates, 'membership_values': term.mf, 'term_name': term_name})
        for fuzzy_variable in list_fuzzy_slow_transition_variable
        for term_name, term in fuzzy_variable.terms.items()
    ])

    # Plot fuzzy epidemic periods with fast transition
    sns.lineplot(df_fuzzy_fast_transition_variable, x='date', y='membership_values', hue='term_name', ax=ax[0],
                 palette='colorblind')
    ax[0].set_xlabel("Month/Year")
    ax[0].set_ylabel("Membership")
    ax[0].set_title('a) Fuzzy variable: fast transition epidemic periods')
    ax[0].legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].tick_params(axis='x', rotation=20)
    ax[0].xaxis.set_major_formatter(myFmt)

    # Plot fuzzy epidemic periods with slow transition
    sns.lineplot(df_fuzzy_slow_transition_variable, x='date', y='membership_values', hue='term_name', ax=ax[1],
                 palette='colorblind')
    ax[1].set_xlabel("Month/Year")
    ax[1].set_ylabel("Membership")
    ax[1].set_title('b) Fuzzy variable: slow transition epidemic periods')
    ax[1].legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].tick_params(axis='x', rotation=20)
    ax[1].xaxis.set_major_formatter(myFmt)

    fig.tight_layout()
    filename = 'images/fuzzy_variable_'+country
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def plot_parameters(df_r0, df_IFR, df_days_to_loss_immunity, country):
    mask_date = mdates.DateFormatter('%m/%Y')
    plt.rc('font', size=6)
    style = dict(color='black')
    sns.set_style("ticks")

    fig, ax = plt.subplots(1, 3, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(4.22)))

    # Plot fuzzy R0
    sns.lineplot(x=df_r0['date'], y=df_r0['r0'], markers=False, color='black', errorbar=('ci', 95), ax=ax[0])
    ax[0].set_ylabel("$R_{0}(t)$")
    ax[0].xaxis.set_major_formatter(mask_date)
    ax[0].set_xlabel('Month/Year')
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_title('a) Time-varying basic reproduction number $R_{0}(t)$')

    # Plot fuzzy IFR
    sns.lineplot(x=df_IFR['date'], y=df_IFR['ifr']*100, markers=False, color='black', errorbar=('ci', 95), ax=ax[1])
    ax[1].set_ylabel("IFR(t) (in %)")
    ax[1].xaxis.set_major_formatter(mask_date)
    ax[1].set_xlabel('Month/Year')
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_title('b) Time-varying Infection Fatality Rate IFR(t)')

    # Plot fuzzy days to loss the immunity
    sns.lineplot(x=df_days_to_loss_immunity['date'], y=df_days_to_loss_immunity['Omega'], markers=False, color='black',
                 errorbar=('ci', 95), ax=ax[2])
    ax[2].set_ylabel("$\Omega(t)$ (in days)")
    ax[2].xaxis.set_major_formatter(mask_date)
    ax[2].set_xlabel('Month/Year')
    ax[2].tick_params(axis='x', rotation=45)
    ax[2].set_title('c) Time-varying days to loss of immunity $\Omega(t)$')

    fig.tight_layout()

    filename = 'images/result_parameters_'+country
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")

    plt.show()