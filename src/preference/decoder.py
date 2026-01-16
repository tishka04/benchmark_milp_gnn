# ==============================================================================
# HIERARCHICAL DECODER FOR PREFERENCE LEARNING
# ==============================================================================
# Wraps the existing FeasibilityDecoder with additional utilities for
# converting EBM decisions to LP worker format
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .data_models import DecisionVector, ScenarioData


@dataclass
class DecodedPlan:
    """
    Decoded operational plan from binary decisions.
    
    Contains both binary commitments and continuous dispatch hints
    that can be passed to the LP worker.
    """
    # Binary decisions [Z, T]
    thermal_on: torch.Tensor
    nuclear_on: torch.Tensor
    battery_charging: torch.Tensor
    battery_discharging: torch.Tensor
    pumped_charging: torch.Tensor
    pumped_discharging: torch.Tensor
    dr_active: torch.Tensor
    import_mode: torch.Tensor  # 1=importing, 0=exporting
    
    # Continuous hints [Z, T] (optional, for warm start)
    thermal_dispatch: Optional[torch.Tensor] = None
    nuclear_dispatch: Optional[torch.Tensor] = None
    battery_charge: Optional[torch.Tensor] = None
    battery_discharge: Optional[torch.Tensor] = None
    pumped_charge: Optional[torch.Tensor] = None
    pumped_discharge: Optional[torch.Tensor] = None
    demand_response: Optional[torch.Tensor] = None
    
    def to_lp_format(self, zone_names: List[str]) -> Dict[str, Dict[Tuple[str, int], float]]:
        """
        Convert to format expected by LP worker.
        
        Returns dict with keys:
            'u_thermal': {(zone, t): value}
            'b_charge_mode': {(zone, t): value}
            'pumped_charge_mode': {(zone, t): value}
            'dr_active': {(zone, t): value}
            'import_mode': {t: value}
        """
        Z, T = self.thermal_on.shape
        
        targets = {
            'u_thermal': {},
            'b_charge_mode': {},
            'pumped_charge_mode': {},
            'dr_active': {},
            'import_mode': {},
        }
        
        for z_idx, zone in enumerate(zone_names[:Z]):
            for t in range(T):
                targets['u_thermal'][(zone, t)] = float(self.thermal_on[z_idx, t].item())
                
                # Battery mode: 1 if charging
                targets['b_charge_mode'][(zone, t)] = float(self.battery_charging[z_idx, t].item())
                
                # Pumped mode: 1 if charging
                targets['pumped_charge_mode'][(zone, t)] = float(self.pumped_charging[z_idx, t].item())
                
                # DR active
                targets['dr_active'][(zone, t)] = float(self.dr_active[z_idx, t].item())
        
        # Import mode is global (first zone or aggregated)
        for t in range(T):
            targets['import_mode'][t] = float(self.import_mode[0, t].item())
        
        return targets
    
    def to_tensor(self) -> torch.Tensor:
        """
        Convert to decision tensor [Z, T, 8].
        
        Feature order:
            0: battery_charging
            1: battery_discharging
            2: pumped_charging
            3: pumped_discharging
            4: dr_active
            5: nuclear_on
            6: thermal_on
            7: import_mode
        """
        return torch.stack([
            self.battery_charging,
            self.battery_discharging,
            self.pumped_charging,
            self.pumped_discharging,
            self.dr_active,
            self.nuclear_on,
            self.thermal_on,
            self.import_mode,
        ], dim=-1)


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical decoder that converts EBM binary decisions to LP worker format.
    
    Optionally includes:
    1. Feasibility correction (mutual exclusion constraints)
    2. Capacity-aware clamping
    3. Temporal consistency smoothing
    """
    
    def __init__(
        self,
        enforce_mutual_exclusion: bool = True,
        enforce_capacity: bool = True,
        smooth_temporal: bool = False,
    ):
        """
        Args:
            enforce_mutual_exclusion: Enforce battery/pumped can't charge and discharge simultaneously
            enforce_capacity: Zero out decisions for assets with zero capacity
            smooth_temporal: Apply temporal smoothing to reduce switching
        """
        super().__init__()
        self.enforce_mutual_exclusion = enforce_mutual_exclusion
        self.enforce_capacity = enforce_capacity
        self.smooth_temporal = smooth_temporal
    
    def forward(
        self,
        u: torch.Tensor,
        scenario: ScenarioData,
    ) -> DecodedPlan:
        """
        Decode binary decisions into operational plan.
        
        Args:
            u: Decision tensor [Z, T, 8] or [B, Z, T, 8]
            scenario: Scenario data with capacities
        
        Returns:
            DecodedPlan with binary commitments
        """
        # Handle batch dimension
        batched = u.dim() == 4
        if not batched:
            u = u.unsqueeze(0)
        
        B, Z, T, F = u.shape
        
        # Extract individual decisions
        battery_charging = u[..., 0]
        battery_discharging = u[..., 1]
        pumped_charging = u[..., 2]
        pumped_discharging = u[..., 3]
        dr_active = u[..., 4]
        nuclear_on = u[..., 5]
        thermal_on = u[..., 6]
        import_mode = u[..., 7]
        
        # Enforce mutual exclusion: can't charge and discharge simultaneously
        if self.enforce_mutual_exclusion:
            # Battery: if both are on, keep the larger one
            batt_conflict = (battery_charging > 0.5) & (battery_discharging > 0.5)
            if batt_conflict.any():
                # Prefer discharge in case of tie
                battery_charging = torch.where(
                    batt_conflict & (battery_charging <= battery_discharging),
                    torch.zeros_like(battery_charging),
                    battery_charging
                )
                battery_discharging = torch.where(
                    batt_conflict & (battery_discharging < battery_charging),
                    torch.zeros_like(battery_discharging),
                    battery_discharging
                )
            
            # Pumped: same logic
            pump_conflict = (pumped_charging > 0.5) & (pumped_discharging > 0.5)
            if pump_conflict.any():
                pumped_charging = torch.where(
                    pump_conflict & (pumped_charging <= pumped_discharging),
                    torch.zeros_like(pumped_charging),
                    pumped_charging
                )
                pumped_discharging = torch.where(
                    pump_conflict & (pumped_discharging < pumped_charging),
                    torch.zeros_like(pumped_discharging),
                    pumped_discharging
                )
        
        # Enforce capacity constraints
        if self.enforce_capacity and scenario is not None:
            # Zero out decisions for assets with zero capacity
            if scenario.thermal_capacity is not None:
                has_thermal = (scenario.thermal_capacity > 0).float().unsqueeze(-1)  # [Z, 1]
                thermal_on = thermal_on * has_thermal
            
            if scenario.nuclear_capacity is not None:
                has_nuclear = (scenario.nuclear_capacity > 0).float().unsqueeze(-1)
                nuclear_on = nuclear_on * has_nuclear
            
            if scenario.battery_power is not None:
                has_battery = (scenario.battery_power > 0).float().unsqueeze(-1)
                battery_charging = battery_charging * has_battery
                battery_discharging = battery_discharging * has_battery
            
            if scenario.pumped_power is not None:
                has_pumped = (scenario.pumped_power > 0).float().unsqueeze(-1)
                pumped_charging = pumped_charging * has_pumped
                pumped_discharging = pumped_discharging * has_pumped
            
            if scenario.dr_capacity is not None:
                has_dr = (scenario.dr_capacity > 0).float().unsqueeze(-1)
                dr_active = dr_active * has_dr
        
        # Binarize
        thermal_on = (thermal_on > 0.5).float()
        nuclear_on = (nuclear_on > 0.5).float()
        battery_charging = (battery_charging > 0.5).float()
        battery_discharging = (battery_discharging > 0.5).float()
        pumped_charging = (pumped_charging > 0.5).float()
        pumped_discharging = (pumped_discharging > 0.5).float()
        dr_active = (dr_active > 0.5).float()
        import_mode = (import_mode > 0.5).float()
        
        # Remove batch dimension if input wasn't batched
        if not batched:
            thermal_on = thermal_on.squeeze(0)
            nuclear_on = nuclear_on.squeeze(0)
            battery_charging = battery_charging.squeeze(0)
            battery_discharging = battery_discharging.squeeze(0)
            pumped_charging = pumped_charging.squeeze(0)
            pumped_discharging = pumped_discharging.squeeze(0)
            dr_active = dr_active.squeeze(0)
            import_mode = import_mode.squeeze(0)
        
        return DecodedPlan(
            thermal_on=thermal_on,
            nuclear_on=nuclear_on,
            battery_charging=battery_charging,
            battery_discharging=battery_discharging,
            pumped_charging=pumped_charging,
            pumped_discharging=pumped_discharging,
            dr_active=dr_active,
            import_mode=import_mode,
        )
    
    def decode_batch(
        self,
        u_batch: torch.Tensor,
        scenarios: List[ScenarioData],
    ) -> List[DecodedPlan]:
        """
        Decode a batch of decisions.
        
        Args:
            u_batch: [B, Z, T, 8] decisions
            scenarios: List of B scenarios
        
        Returns:
            List of B DecodedPlans
        """
        B = u_batch.shape[0]
        plans = []
        
        for i in range(B):
            plan = self.forward(u_batch[i], scenarios[i] if i < len(scenarios) else None)
            plans.append(plan)
        
        return plans


def decision_to_decoder_output(
    u: torch.Tensor,
    n_output_features: int = 13,
) -> torch.Tensor:
    """
    Convert EBM decision format [Z, T, 8] to GNN decoder output format [Z*T, F].
    
    The LP worker expects decoder output in a specific format.
    This function bridges the two representations.
    
    Args:
        u: Decision tensor [Z, T, 8]
        n_output_features: Number of output features expected by LP worker
    
    Returns:
        decoder_output: [Z*T, F] tensor compatible with LP worker
    """
    Z, T, _ = u.shape
    
    # Map from EBM features [8] to decoder features [F]
    # EBM: [batt_ch, batt_dis, pump_ch, pump_dis, dr, nuclear, thermal, import_mode]
    # LP worker expects: different order and additional features
    
    decoder_out = torch.zeros(Z, T, n_output_features, device=u.device)
    
    # Basic mapping (adjust indices as needed for your LP worker)
    decoder_out[..., 0] = u[..., 6]  # thermal_on
    decoder_out[..., 1] = u[..., 5]  # nuclear_on
    decoder_out[..., 4] = u[..., 4]  # dr_active
    decoder_out[..., 7] = u[..., 0]  # battery_charging
    decoder_out[..., 8] = u[..., 1]  # battery_discharging
    decoder_out[..., 9] = u[..., 2]  # pumped_charging
    decoder_out[..., 10] = u[..., 3]  # pumped_discharging
    decoder_out[..., 11] = u[..., 7]  # import_mode
    
    return decoder_out
