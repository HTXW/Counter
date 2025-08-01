"""A collection of helper functions for generating the results of the
simulation.

"""

import pickle
from datetime import datetime
import pprint
import pandas as pd
import matplotlib.pyplot as plt

from philharmonic import conf
import philharmonic as ph
from philharmonic.logger import *
from philharmonic.scheduler import evaluator
from philharmonic.utils import loc
from philharmonic import Schedule

def pickle_results(schedule):
    schedule.actions.to_pickle(loc('schedule.pkl'))

ev = evaluator

"""def generate_series_results(cloud, env, schedule, nplots):
    generate power of IT equipment (power) and power of IT equipment
    including the cooling overhead time series for the simulation
    duration.
    
    info('\nDynamic results\n---------------')
    util, power, power_total, freq = ev.calculate_components(
        cloud, env, schedule, env.el_prices, env.temperature,
        power_model=conf.power_model)

    info('Utilisation (%)')
    info(str(util * 100))
    # print('- weighted mean per no')
    # weighted_mean(util[util>0])
    # util[util>0].mean().dropna().mean() * 100
    # TODO: maybe weighted mean for non-zero util
    # ax = plt.subplot(nplots, 1, 1)
    # ax.set_title('Utilisation (%)')
    # util.plot(ax=ax)

    # frequency & power
    #-----------------
    if freq is not None:
        info('\nPM frequencies (MHz)')
        info(freq.describe())

    if conf.save_power:
        power.to_pickle(loc('power.pkl'))
    ax = plt.subplot(nplots, 1, 3)
    ax.set_title('Computational power (W)')
    power.plot(ax=ax)
    energy = ph.joul2kwh(ph.calculate_energy(power))
    info('IT Equipment Power (W)')
    info(power.describe())
    info('Cooling Power (W)')
    info(power_total.describe())
    # info('\nEnergy (kWh)')
    # info(energy)
    # info(' - total:')
    # info(energy.sum())

    # cooling overhead
    #-----------------
    ax = plt.subplot(nplots, 1, 4)
    ax.set_title('Total power (W)')
    power_total.plot(ax=ax)
    if conf.save_power:
        power_total.to_pickle(loc('power_total.pkl'))
    energy_total = ph.joul2kwh(ph.calculate_energy(power_total))
    # info('\nEnergy with cooling (kWh)')
    # info(energy_total)
    # info(' - total:')
    # info(energy_total.sum())

    # PM utilisation
    #---------------
    #info('\nPM Avg.Utilization')
    #info(util.mean())
    info('\nPM Utilization')
    info(util.describe())

    info('\nMax of Avg. PM Utilization')
    info(util.mean().max())


    # mean utilization
    info('\nAvg.Utilization')
    info(util.mean().mean())

    info('\nMax Utilization')
    info(util.max().max())"""


def generate_series_results(cloud, env, schedule, nplots):
    """Generate power of IT equipment (power) and power of IT equipment
    including the cooling overhead time series for the simulation duration.
    """
    info('\nDynamic results\n---------------')
    util, power, power_total, freq = ev.calculate_components(
        cloud, env, schedule, env.el_prices, env.temperature,
        power_model=conf.power_model
    )

    info('Utilisation (%)')
    info(str(util * 100))

    # Frequency and power details
    if freq is not None:
        info('\nPM frequencies (MHz)')
        info(freq.describe())

    if conf.save_power:
        power.to_pickle(loc('power.pkl'))

    ax = plt.subplot(nplots, 1, 3)
    ax.set_title('Computational power (W)')
    power.plot(ax=ax)
    energy = ph.joul2kwh(ph.calculate_energy(power))
    info('IT Equipment Power (W)')
    info(power.describe())
    info('Cooling Power (W)')
    info(power_total.describe())

    ax = plt.subplot(nplots, 1, 4)
    ax.set_title('Total power (W)')
    power_total.plot(ax=ax)
    if conf.save_power:
        power_total.to_pickle(loc('power_total.pkl'))

    energy_total = ph.joul2kwh(ph.calculate_energy(power_total))
    info('\nEnergy with cooling (kWh)')
    info(energy_total)
    info(' - total:')
    info(energy_total.sum())

    # PM utilisation and average utilisation
    info('\nPM Utilization')
    info(util.describe())
    info('\nMax of Avg. PM Utilization')
    info(util.mean().max())
    info('\nAvg.Utilization')
    info(util.mean().mean())
    info('\nMax Utilization')
    info(util.max().max())
# TODO: split into smaller functions
def serialise_results(cloud, env, schedule):
    fig = plt.figure(1)#, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.2, top=0.9, hspace=0.5)

    nplots = 4
    pickle_results(schedule)
    cloud.reset_to_initial()
    info('Simulation timeline\n-------------------')
    evaluator.print_history(cloud, env, schedule)

    # geotemporal inputs
    #-------------------
    ax = plt.subplot(nplots, 1, 1)
    ax.set_title('Electricity prices ($/kWh)')
    env.el_prices.plot(ax=ax)

    if env.temperature is not None:
        ax = plt.subplot(nplots, 1, 2)
        ax.set_title('Temperature (C)')
        env.temperature.plot(ax=ax)

    # dynamic results
    #----------------
    generate_series_results(cloud, env, schedule, nplots)

    # the values used for the aggregated results
    energy = evaluator.combined_energy(cloud, env, schedule)
    energy_total = evaluator.combined_energy(cloud, env, schedule,
                                             env.temperature)

    # Aggregated results
    #===================
    info('\nAggregated results\n------------------')

    # migration overhead
    #-------------------
    migration_energy, migration_cost = evaluator.calculate_migration_overhead(
        cloud, env, schedule
    )
    info('Migration energy (kWh)')
    info(migration_energy)
    info(' - total with migrations:')
    info(energy_total + migration_energy)
    info('\nMigration cost ($)')
    info(migration_cost)

    # electricity costs
    #------------------
    # TODO: update the dynamic cost calculations to work on the new power model
    # TODO: reenable
    # en_cost_IT = evaluator.calculate_cloud_cost(power, env.el_prices)
    info('\nElectricity costs ($)')
    # info(' - electricity cost without cooling:')
    # info(en_cost_IT)
    info(' - total electricity cost without cooling:')
    en_cost_IT_total = evaluator.combined_cost(cloud, env, schedule,
                                               env.el_prices,
                                               power_model=conf.power_model)
    info(en_cost_IT_total)

    # TODO: reenable
    # en_cost_with_cooling = evaluator.calculate_cloud_cost(power_total,
    #                                                       env.el_prices)
    # info(' - electricity cost with cooling:')
    # info(en_cost_with_cooling)
    info(' - total electricity cost with cooling:')
    en_cost_with_cooling_total = evaluator.combined_cost(
        cloud, env, schedule, env.el_prices, env.temperature,
        power_model=conf.power_model
    )
    info(en_cost_with_cooling_total)
    info(' - total electricity cost with migrations:')
    en_cost_combined = en_cost_with_cooling_total + migration_cost
    info(en_cost_combined)

    # the schedule if we did not apply any frequency scaling
    schedule_unscaled = Schedule()
    schedule_unscaled.actions = schedule.actions[
        schedule.actions.apply(lambda a : not a.name.endswith('freq'))
    ]

    # QoS aspects
    info(' - total profit from users:')
    serv_profit = evaluator.calculate_service_profit(cloud, env, schedule)
    info(f'${serv_profit}')
    info(' - profit loss due to scaling:')
    serv_profit_unscaled = evaluator.calculate_service_profit(
        cloud, env, schedule_unscaled
    )
    scaling_profit_loss = serv_profit_unscaled - serv_profit
    scaling_profit_loss_rel = scaling_profit_loss / serv_profit_unscaled
    info(f'${scaling_profit_loss}')
    info(f'{scaling_profit_loss_rel:.2%}')

    # frequency savings
    info(' - frequency scaling savings (compared to no scaling):')
    en_cost_combined_unscaled = evaluator.combined_cost (
        cloud, env, schedule_unscaled, env.el_prices, env.temperature,
        power_model=conf.power_model) + migration_cost

    scaling_savings_abs = en_cost_combined_unscaled - en_cost_combined
    info(f'${scaling_savings_abs}')
    scaling_savings_rel = scaling_savings_abs / en_cost_combined_unscaled
    info(f'{scaling_savings_rel:.2%}')

    # Capacity constraints
    #---------------------
    # TODO: these two

    # aggregated results
    aggregated = [energy, en_cost_IT_total,
                  energy_total + migration_energy, en_cost_combined,
                  serv_profit, serv_profit - en_cost_combined]
    aggr_names = ['IT energy (kWh)', 'IT cost ($)',
                  'Total energy (kWh)', 'Total cost ($)',
                  'Service revenue ($)', 'Gross profit ($)']
    # http://en.wikipedia.org/wiki/Gross_profit
    # Towards Profitable Virtual Machine Placement in the Data Center Shi
    # and Hong 2011 - total profit, revenue and operational cost
    aggregated_results = pd.Series(aggregated, aggr_names)
    aggregated_results.to_pickle(loc('results.pkl'))
    aggregated_results.to_csv(loc('results.csv'))
    #aggregated_results.plot(kind='bar')
    info('\n')
    info(aggregated_results)


    if conf.liveplot:
        plt.show()
    elif conf.liveplot:
        plt.savefig(loc('results-graph.pdf'))

    info(f'\nDone. Results saved to: {conf.output_folder}')
    return aggregated_results
