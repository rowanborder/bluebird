#!/usr/bin/env python3
"""
pi5_power_monitor.py

Monitors Raspberry Pi 5 power consumption using vcgencmd pmic_read_adc
and estimates battery runtime for a 50Wh battery.

Usage:
  python pi5_power_monitor.py [--interval SECONDS] [--battery-wh WH]
"""

import argparse
import subprocess
import time
import re
from collections import deque
from datetime import timedelta


def read_pmic_adc():
    """
    Read PMIC ADC values using vcgencmd.
    Returns a dictionary with voltage and current measurements.
    """
    try:
        result = subprocess.run(
            ['vcgencmd', 'pmic_read_adc'],
            capture_output=True,
            text=True,
            check=True
        )
        
        data = {}
        for line in result.stdout.strip().split('\n'):
            # Parse lines like "3V7_WL_SW_A current(0)=0.09564114A"
            # or "3V7_WL_SW_V volt(8)=3.64038700V"
            current_match = re.match(r'\s*(\S+)\s+current\(\d+\)=([\d.]+)A', line)
            volt_match = re.match(r'\s*(\S+)\s+volt\(\d+\)=([\d.]+)V', line)
            
            if current_match:
                name, value = current_match.groups()
                data[name] = data.get(name, {})
                data[name]['current'] = float(value)
            elif volt_match:
                name, value = volt_match.groups()
                data[name] = data.get(name, {})
                data[name]['voltage'] = float(value)
        
        return data
    except subprocess.CalledProcessError as e:
        print(f"Error reading PMIC ADC: {e}")
        return None
    except FileNotFoundError:
        print("Error: vcgencmd not found. Are you running this on a Raspberry Pi?")
        return None


def calculate_power(pmic_data):
    """
    Calculate power consumption from PMIC data.
    Uses VDD_CORE rail as the primary power measurement.
    Returns power in Watts for each rail and total.
    """
    power_sources = []
    vdd_core_power = None
    
    # Check if we have VDD_CORE measurements
    if 'VDD_CORE_V' in pmic_data and 'VDD_CORE_A' in pmic_data:
        vdd_core_v = pmic_data['VDD_CORE_V'].get('voltage')
        vdd_core_a = pmic_data['VDD_CORE_A'].get('current')
        
        if vdd_core_v is not None and vdd_core_a is not None:
            vdd_core_power = vdd_core_v * vdd_core_a
            power_sources.append({
                'name': 'VDD_CORE',
                'voltage': vdd_core_v,
                'current': vdd_core_a,
                'power': vdd_core_power
            })
    
    # Add other rails for reference
    voltage_rails = {}
    current_rails = {}
    
    for name, measurements in pmic_data.items():
        if name.endswith('_V'):
            base_name = name[:-2]
            if 'voltage' in measurements:
                voltage_rails[base_name] = measurements['voltage']
        elif name.endswith('_A'):
            base_name = name[:-2]
            if 'current' in measurements:
                current_rails[base_name] = measurements['current']
    
    # Match voltage and current pairs (excluding VDD_CORE which we already added)
    for base_name in voltage_rails:
        if base_name == 'VDD_CORE':
            continue
        if base_name in current_rails:
            voltage = voltage_rails[base_name]
            current = current_rails[base_name]
            power = voltage * current
            
            # Only include rails with measurable power (> 0.001W)
            if power > 0.001:
                power_sources.append({
                    'name': base_name,
                    'voltage': voltage,
                    'current': current,
                    'power': power
                })
    
    # Sort by power consumption (highest first)
    power_sources.sort(key=lambda x: x['power'], reverse=True)
    
    return power_sources


def format_time(hours):
    """Format hours as a human-readable string."""
    if hours < 0:
        return "N/A"
    
    td = timedelta(hours=hours)
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor Raspberry Pi 5 power consumption and estimate battery runtime'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Measurement interval in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--battery-wh',
        type=float,
        default=50.0,
        help='Battery capacity in Watt-hours (default: 50.0)'
    )
    parser.add_argument(
        '--avg-window',
        type=int,
        default=30,
        help='Number of samples for moving average (default: 30)'
    )
    
    args = parser.parse_args()
    
    print(f"Raspberry Pi 5 Power Monitor")
    print(f"{'='*60}")
    print(f"Battery capacity: {args.battery_wh} Wh")
    print(f"Measurement interval: {args.interval}s")
    print(f"Averaging window: {args.avg_window} samples")
    print(f"{'='*60}\n")
    
    # Store power measurements for averaging
    power_history = deque(maxlen=args.avg_window)
    
    try:
        while True:
            # Read PMIC data
            pmic_data = read_pmic_adc()
            
            if pmic_data is None:
                time.sleep(args.interval)
                continue
            
            # Calculate power from each source
            power_sources = calculate_power(pmic_data)
            
            if not power_sources:
                print("No power sources detected")
                time.sleep(args.interval)
                continue
            
            # Calculate total power
            total_power = sum(ps['power'] for ps in power_sources)
            power_history.append(total_power)
            
            # Calculate average power
            avg_power = sum(power_history) / len(power_history)
            
            # Estimate battery runtime
            if avg_power > 0:
                runtime_hours = args.battery_wh / avg_power
                runtime_str = format_time(runtime_hours)
            else:
                runtime_str = "N/A"
            
            # Display current readings
            # Display current readings
            print(f"\033[2J\033[H", end='')  # Clear screen and move to top
            print(f"Raspberry Pi 5 Power Monitor - {time.strftime('%H:%M:%S')}")
            print(f"{'='*70}")
            
            # Display individual rails
            print(f"\n{'Rail':<15} {'Voltage':>10} {'Current':>10} {'Power':>10}")
            print(f"{'-'*50}")
            for ps in power_sources:
                print(f"{ps['name']:<15} {ps['voltage']:>9.3f}V {ps['current']:>9.3f}A {ps['power']:>9.3f}W")
            
            # Display summary
            print(f"\n{'='*70}")
            print(f"Total Power:      {total_power:>6.3f}W")
            print(f"Average Power:    {avg_power:>6.3f}W (over {len(power_history)} samples)")
            if power_history and len(power_history) > 1:
                print(f"Min Power:        {min(power_history):>6.3f}W")
                print(f"Max Power:        {max(power_history):>6.3f}W")
            print(f"Battery Capacity: {args.battery_wh:>6.1f}Wh")
            print(f"Est. Runtime:     {runtime_str}")
            print(f"{'='*70}")
            print(f"\nPress Ctrl+C to stop...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        
        if power_history:
            avg_power = sum(power_history) / len(power_history)
            print(f"\nFinal Statistics:")
            print(f"  Average Power: {avg_power:.3f}W")
            print(f"  Min Power: {min(power_history):.3f}W")
            print(f"  Max Power: {max(power_history):.3f}W")
            
            if avg_power > 0:
                runtime_hours = args.battery_wh / avg_power
                print(f"  Estimated Runtime: {format_time(runtime_hours)} on {args.battery_wh}Wh battery")


if __name__ == '__main__':
    main()
