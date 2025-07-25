"""The philharmonic simulator.
Traces geotemporal input data, asks the scheduler to determine actions
and simulates the outcome of the schedule."""

import pickle
from datetime import datetime
import pandas as pd
import pprint
import philharmonic as ph

from philharmonic import conf, Schedule
from philharmonic.logger import *
from .import inputgen
from .results import serialise_results
from philharmonic.manager.imanager import IManager
from philharmonic.utils import loc, common_loc, input_loc
from philharmonic.scheduler.generic.fbf_optimiser import FBFOptimiser
from philharmonic.scheduler import NoScheduler, FBFScheduler, BFDScheduler
from philharmonic.scheduler.peak_pauser.peak_pauser import PeakPauser
from philharmonic.scheduler.GGCNNBasedScheduler import GGCNNBasedScheduler
from philharmonic.cloud.driver.simdriver import simdriver
from philharmonic.cloud.driver.nodriver import nodriver
from .environment import SimulatedEnvironment, PPSimulatedEnvironment, FBFSimpleSimulatedEnvironment, GASimpleSimulatedEnvironment


if conf.plot_on_server:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


# old scheduler design...
#-------------------------

def geotemporal_inputs():
    """Read time series for el. prices and temperatures
    at different locations.
    """
    info(" - reading geotemporal inputs")
    freq = 'H'
    # el. prices
    el_prices_pth = 'io/geotemp/el_prices-usa.pkl'
    el_prices = pd.read_pickle(el_prices_pth)
    # - resample to desired freqency
    el_prices = el_prices.resample(freq)
    debug(str(el_prices))

    # temperatures
    temperatures_pth = 'io/geotemp/temperature-usa.pkl'
    temperatures = pd.read_pickle(temperatures_pth)
    temperatures = temperatures.resample(freq)
    debug(str(temperatures))
    # common index is actually in temperatures (subset of prices)

    return el_prices, temperatures


def server_locations(servers, possible_locations):
    """Change servers by setting a location."""
    #Todo: Potentially separate into DCs
    for i, s in enumerate(servers):
        s.loc = possible_locations[i]


def VM_requests(start, end):
    return inputgen.normal_vmreqs(start, end)


def prepare_known_data(dataset, t, future_horizon=None): # TODO: use pd.Panel for dataset
    """ @returns a subset of the @param dataset
    (a tuple of pd.Series objects)
    that is known at moment @param t
    """
    future_horizon = future_horizon or pd.offsets.Hour(4)
    el_prices, temperatures = dataset # unpack
    # known data (past and future up to a point)
    known_el_prices = el_prices[:t+future_horizon]
    known_temperatures = temperatures[:t+future_horizon]
    return known_el_prices, known_temperatures

#TODO: - shorthand to access temp, price in server
# new simulator design

class Simulator(IManager):
    # simulate the passage of time, prepare all data for scheduler
    factory = {
        "scheduler": "GGCNNBasedScheduler",
        "environment": "FBFSimpleSimulatedEnvironment",
        "cloud": "peak_pauser_infrastructure",
        "driver": "simdriver",
        "times": "two_days",
        "requests": None,  # inputgen.normal_vmreqs,
        "servers": None,  # inputgen.small_infrastructure,
        "el_prices": "simple_el",
        "temperature": "simple_temperature",
    }
    # GGCNNBasedScheduler
    def __init__(self, factory=None, custom_scheduler=None):
        # Initialize Simulator class from IManager
        if factory:
            self.factory = factory
        if custom_scheduler:
            self.custom_scheduler = custom_scheduler
        super(Simulator, self).__init__()

        # Instantiate the environment class
        environment_class = {
            "SimulatedEnvironment": SimulatedEnvironment,
            "PPSimulatedEnvironment": PPSimulatedEnvironment,
            "FBFSimpleSimulatedEnvironment": FBFSimpleSimulatedEnvironment,
            "GASimpleSimulatedEnvironment": GASimpleSimulatedEnvironment
        }.get(self.factory['environment'], SimulatedEnvironment)

        self.environment = environment_class(times=conf.times)

        # Load environment data
        self.environment.el_prices = self._create(inputgen, self.factory['el_prices'])
        self.environment.temperature = self._create(inputgen, self.factory['temperature'])
        SD_el = self.factory.get('SD_el', 0)
        SD_temp = self.factory.get('SD_temp', 0)
        self.environment.model_forecast_errors(SD_el, SD_temp)
        self.real_schedule = Schedule()

        self.cloud = self._create(inputgen, self.factory['cloud'])
        self.scheduler = self._initialize_scheduler()
        self.requests = self._initialize_requests()
        self.driver = self._initialize_driver()

    def _initialize_driver(self):
        driver_dict = {
            "simdriver": simdriver,
            "nodriver": nodriver
        }

        driver_class = driver_dict.get(self.factory['driver'])
        if driver_class is None:
            raise ValueError(f"Unknown driver type: {self.factory['driver']}")
        return driver_class()

    def _initialize_scheduler(self):
        scheduler_dict = {
            "GGCNNBasedScheduler": GGCNNBasedScheduler,
            "PeakPauser": PeakPauser,
            "NoScheduler": NoScheduler,
            "FBFScheduler": FBFOptimiser,
            "BFDScheduler": BFDScheduler,
        }

        scheduler_class = scheduler_dict.get(self.factory['scheduler'])
        return scheduler_class(cloud=self.cloud, driver=self, environment=self.environment)

    def _initialize_requests(self):
        times = self.factory.get('times')
        if times:
            start, end = times[0], times[-1]
            offset = self.factory.get('requests_offset')
            requests_kwargs = {'start': start, 'end': end, 'offset': offset} if offset else {'start': start, 'end': end}
            return self._create(inputgen, self.factory['requests'], **requests_kwargs)
        else:
            return None

    def apply_actions(self, actions):
        # self.cloud.reset_to_real()
        for t, action in actions.items():
            self.cloud.apply_real(action)
            self.real_schedule.add(action, t)
            self.driver.apply_action(action, t)
            # Log the current state
            state = self.cloud.get_current()
            info(f"After applying action at {t}:")
            info(f"VMs in cloud: {state.vms}")
            info(f"VM allocations: {state.alloc}")

    def prompt(self):
        if conf.prompt_show_cloud:
            if conf.prompt_ipdb:
                import ipdb; ipdb.set_trace()
            else:
                input('Press enter to continue.')

    def show_cloud_usage(self):
        self.cloud.show_usage()
        self.prompt()

    def run(self, steps=5):
        if conf.show_cloud_interval is not None:
            t_show = conf.start + conf.show_cloud_interval

        self.scheduler.initialize()
        passed_steps = 0
        for t in self.environment.itertimes():
            debug('-' * 25 + '\n| t={} |\n'.format(t) + '-' * 25)
            passed_steps += 1
            if steps is not None and passed_steps > steps:
                break

            # Get requests & update model
            requests = self.environment.get_requests()
            self.apply_actions(requests)

            # Call scheduler to decide on actions
            schedule = self.scheduler.reevaluate()
            # self.cloud.reset_to_real()

            period = self.environment.get_period()
            actions = schedule.filter_current_actions(t, period)
            if len(actions) > 0:
                debug('Applying actions at time {}:\n{}\n'.format(t, actions))
                self.apply_actions(actions)

            if conf.show_cloud_interval is not None and t == t_show:
                t_show = t_show + conf.show_cloud_interval
                self.show_cloud_usage()

        return self.cloud, self.environment, self.real_schedule


def run(self, steps=5):
    """Run the simulation by iterating through times, reevaluating schedules, and simulating actions."""
    if conf.show_cloud_interval is not None:
        t_show = conf.start + conf.show_cloud_interval

    self.scheduler.initialize()
    passed_steps = 0
    for t in self.environment.itertimes():
        debug('-' * 25 + '\n| t={} |\n'.format(t) + '-' * 25)
        passed_steps += 1
        if steps is not None and passed_steps > steps:
            break

        # Get requests & update the model
        requests = self.environment.get_requests()
        self.apply_actions(requests)

        # Call scheduler to decide on actions
        schedule = self.scheduler.reevaluate()
        self.cloud.reset_to_real()

        period = self.environment.get_period()
        actions = schedule.filter_current_actions(t, period)
        if len(actions) > 0:
            debug('Applying:\n{}\n'.format(actions))
        self.apply_actions(actions)

        if conf.show_cloud_interval is not None and t == t_show:
            t_show = t_show + conf.show_cloud_interval
            self.show_cloud_usage()

    return self.cloud, self.environment, self.real_schedule


# TODO: these other simulator subclasses should not be necessary
class PeakPauserSimulator(Simulator):
    def __init__(self, factory=None):
        if factory is not None:
            self.factory = factory
        self.factory["scheduler"] = "PeakPauser"
        self.factory["environment"] = "PPSimulatedEnvironment"
        super(PeakPauserSimulator, self).__init__()

    def run(self): # TODO: use Simulator.run instead
        """go through all the timesteps and call the scheduler to ask for
        actions
        """
        self.environment.times = list(range(24))
        self.environment._period = pd.offsets.Hour(1)
        self.scheduler.initialize()
        for hour in self.environment.times:
            # TODO: set time in the environment instead of here
            timestamp = pd.Timestamp('2013-02-20 {0}:00'.format(hour))
            self.environment.set_time(timestamp)
            # call scheduler to create new cloud state (if an action is made)
            schedule = self.scheduler.reevaluate()
            # TODO: when an action is applied to the current state, forward it
            # to the driver as well
            period = self.environment.get_period()
            actions = schedule.filter_current_actions(timestamp, period)
            self.apply_actions(actions)
        # TODO: use schedule instance
        # events = self.cloud.driver.events


class FBFSimulator(Simulator):
    def __init__(self, factory=None):
        if factory is not None:
            self.factory = factory
        self.factory["scheduler"] = "FBFScheduler"
        self.factory["environment"] = "FBFSimpleSimulatedEnvironment"
        super(FBFSimulator, self).__init__()


class NoSchedulerSimulator(Simulator):
    def __init__(self):
        self.factory["scheduler"] = "NoScheduler"
        super(NoSchedulerSimulator, self).__init__()


# -- common functions --------------------------------

def log_config_info(simulator):
    """Log the essential configuration information."""
    info(f'- output_folder: {conf.output_folder}')
    if conf.factory["times"] == "times_from_conf":
        info(f'- times: {conf.start} - {conf.end}')
    if conf.factory["el_prices"] == "el_prices_from_conf":
        info(f'- el_price_dataset: {conf.el_price_dataset}')
    if conf.factory["temperature"] == "temperature_from_conf":
        info(f'- temperature_dataset: {conf.temperature_dataset}')
    info('- forecasting:')
    info(f'  * periods: {conf.factory["forecast_periods"]}')
    info(f'  * errors: SD_el={conf.factory["SD_el"]}, SD_temp={conf.factory["SD_temp"]}')
    info(f'- power_model: {conf.power_model}')
    info('\n- scheduler: {}'.format(conf.factory['scheduler']))
    if conf.factory['scheduler_conf'] is not None:
        info('  * conf: \n{}'.format(
            pprint.pformat(conf.factory['scheduler_conf'])
        ))

    info(f'\nServers {common_loc("workload/servers.pkl")}'
         f'-> will copy to: {os.path.relpath(input_loc("servers.pkl"))}'
         f'\n-------\n{simulator.cloud.servers}'
        # pprint.pformat(simulator.cloud.servers)
        # simulator.cloud.show_usage()
    )
    if conf.power_freq_model is not False:
        info(f'\n- freq. scale from {conf.freq_scale_min} to {conf.freq_scale_max} by {conf.freq_scale_delta}.')
    info(f'\nRequests ({common_loc("workload/requests.pkl")} -> will copy to: {os.path.relpath(input_loc("requests.pkl"))})'
         f'\n--------\n{simulator.requests}\n')

    if conf.prompt_configuration:
        prompt_res = input('Config good? Press enter to continue...')


def archive_inputs(simulator):
    """copy input files together with the results (for archive reasons)"""
    with open(input_loc('servers.pkl'), 'wb') as pkl_srv:
        pickle.dump(simulator.cloud, pkl_srv)
    simulator.requests.to_pickle(input_loc('requests.pkl'))


def before_start(simulator):
    log_config_info(simulator)
    archive_inputs(simulator)

# -- simulation starter ------------------------------

# schedule.py routes straight to here

def run(steps=None, custom_scheduler=None):
    """Run the simulation."""
    info('\nSETTINGS\n########\n')

    # create simulator from the conf
    # -------------------------------
    simulator = Simulator(conf.get_factory(), custom_scheduler)
    before_start(simulator)

    # run the simulation
    # -------------------
    info('\nSIMULATION\n##########\n')
    start_time = datetime.now()
    info(f'Simulation started at time: {start_time}')
    cloud, env, schedule = simulator.run(steps)
    info('RESULTS\n#######\n')

    # serialise and log the results
    # ------------------------------
    results = serialise_results(cloud, env, schedule)

    end_time = datetime.now()
    info(f'Simulation finished at time: {end_time}')
    info(f'Duration: {end_time - start_time}\n')
    return results


if __name__ == "__main__":
    run()