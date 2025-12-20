from dataclasses import dataclass

@dataclass(frozen=True)
class ThermalDefaults:
    unit_capacity_mw: float = 120.0
    min_power_fraction: float = 0.45
    startup_cost: float = 1200.0
    emission_rate_t_per_mwh: float = 0.72

@dataclass(frozen=True)
class SolarDefaults:
    unit_capacity_mw: float = 45.0

@dataclass(frozen=True)
class WindDefaults:
    unit_capacity_mw: float = 80.0

@dataclass(frozen=True)
class NuclearDefaults:
    unit_capacity_mw: float = 900.0
    min_power_fraction: float = 0.20  # 20% minimum when ON (must-run baseload)
    fuel_cost_per_mwh: float = 1.0

@dataclass(frozen=True)
class StorageDefaults:
    power_per_unit_mw: float = 45.0
    energy_hours: float = 4.0
    efficiency: float = 0.90

@dataclass(frozen=True)
class HydroReservoirDefaults:
    power_per_unit_mw: float = 220.0
    energy_hours: float = 6.0
    spill_cost: float = 5.0

@dataclass(frozen=True)
class RunOfRiverDefaults:
    output_per_unit_mw: float = 60.0

@dataclass(frozen=True)
class PumpedStorageDefaults:
    power_per_unit_mw: float = 80.0
    energy_hours: float = 8.0
    efficiency: float = 0.87

@dataclass(frozen=True)
class TransmissionDefaults:
    base_capacity_mw: float = 220.0

THERMAL = ThermalDefaults()
SOLAR = SolarDefaults()
WIND = WindDefaults()
NUCLEAR = NuclearDefaults()
BATTERY = StorageDefaults()
HYDRO_RES = HydroReservoirDefaults()
HYDRO_ROR = RunOfRiverDefaults()
PUMPED = PumpedStorageDefaults()
TRANS = TransmissionDefaults()

VALUE_OF_LOST_LOAD = 15000.0
DEMAND_RESPONSE_COST = 1500.0
RENEWABLE_SPILL_COST = 8.0
IMPORT_BASE_CAPACITY = 320.0
