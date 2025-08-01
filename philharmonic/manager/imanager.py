import copy
import importlib
import sys
import os

from philharmonic import conf
from philharmonic import scheduler
from philharmonic.simulator import inputgen
from philharmonic.cloud import driver
from philharmonic.simulator import environment
from philharmonic import logger
from philharmonic.utils import deprecated
from philharmonic.scheduler import PeakPauser, NoScheduler
from philharmonic.scheduler.GGCNNBasedScheduler import GGCNNBasedScheduler  # Import new scheduler


class IManager:
    """abstract cloud manager. Asks the scheduler what to do, given the current
    state of the environment and arbitrates the actions to the cloud."""

    factory = {
        "scheduler": None,
        "environment": None,
        "cloud": None,
        "driver": None,
        "times": None,
        "requests": None,
    }

    @classmethod
    def factory_copy(cls):
        return copy.copy(cls.factory)

    def _empty(self, *args, **kwargs):
        return None

    def _create_old(self, cls, *args, **kwargs):
        return (cls or self._empty)(*args, **kwargs)

    def _create(self, module, cls, *args, **kwargs):
        """Find class called @param cls, defined in @param module
        and instantiate, passing the remaining arguments to it.
        """
        if cls is None:
            return None
        else:
            return getattr(module, cls)(*args, **kwargs)

    def arm(self):
        """Take assembled components and inter-connect them."""
        # Arm driver - TODO if necessary

        # Arm scheduler
        if self.scheduler:
            self.scheduler.cloud = self.cloud
            self.scheduler.environment = self.environment
            # Additional setup for GGCNNBasedScheduler
            if isinstance(self.scheduler, GGCNNBasedScheduler):
                # Initialize any specific attributes or configurations
                self.scheduler.initialize_specifics()

        # Arm cloud.driver
        # self.cloud.driver.environment = self.environment

        # Apply scheduler-specific configurations
        try:
            for key, value in self.scheduler_conf.items():
                setattr(self.scheduler, key, value)
        except AttributeError:
            pass



    def run(self):
        raise NotImplemented


class ManagerFactory:
    """Easier manager creation"""
    @staticmethod
    @deprecated

    def __init__(self, factory=None):
        """Create a manager's assets.
        @param factory: optional dict of components to use.
        """
        if not factory:
            factory = self.factory
        # we getattr/import_module from strings, so that we don't have to
        # import classes/functions in the conf module directly.
        try:
            # If `custom_scheduler` was provided, use it
            if hasattr(self, 'custom_scheduler') and self.custom_scheduler:
                custom_scheduler = self.custom_scheduler
                logger.debug(custom_scheduler)
                sys.path.append(os.getcwd())
                self.scheduler = importlib.import_module(custom_scheduler)
                # TODO: get the class and instantiate it
            else:
                # Use the scheduler from the config
                self.scheduler = self._create(scheduler, factory['scheduler'])
        except AttributeError:
            # Handle any exceptions related to scheduler creation
            self.scheduler = self._create(scheduler, factory['scheduler'])

        self.cloud = self._create(inputgen, factory['cloud'])
        if factory['driver'] is not None:
            # self.driver = getattr(driver, factory['driver'])
            self.driver = importlib.import_module('.' + factory['driver'],
                                                  'philharmonic.cloud.driver')
        else:
            self.driver = None

        self.times = self._create(inputgen, factory['times'])
        if self.times is not None:
            requests_kwargs = {'start': self.times[0], 'end': self.times[-1]}
            if ('requests_offset' in factory and
                    factory['requests_offset'] is not None):
                requests_kwargs['offset'] = factory['requests_offset']
            self.requests = self._create(inputgen,
                                         factory['requests'],
                                         **requests_kwargs)
        else:
            self.requests = None
        try:
            periods = factory['forecast_periods']
        except:
            self.environment = self._create(environment,
                                            factory['environment'],
                                            self.times, self.requests)
        else:
            self.environment = self._create(environment,
                                            factory['environment'],
                                            self.times, self.requests,
                                            forecast_periods=periods)
        self.arm()

    def create_from_conf(conf):
        """Pass a conf module to read parameters from. The method creates
        a scheduler instance and constructs a manager with it.
        The manager's constructor will link it with the appropriate
        environment and cloud objects."""

        # Managers to choose from
        from philharmonic.manager.manager import Manager
        from philharmonic.simulator.simulator import Simulator

        # Create the scheduler
        ChosenScheduler = locals().get(conf.scheduler, NoScheduler)  # Dynamically get the chosen scheduler
        scheduler = ChosenScheduler() if ChosenScheduler else NoScheduler()

        # Connect everything in a manager
        ChosenManager = locals()[conf.manager]
        manager = ChosenManager(scheduler)
        return manager
