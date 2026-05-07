"""
WebSocket utilities for broadcasting progress updates.
"""

import asyncio
import json
from datetime import datetime
from channels.layers import get_channel_layer


async def broadcast_simulation_progress(simulation_id, message, iteration=0, status='processing'):
    """
    Broadcast a progress update to all connected WebSocket clients for a simulation.
    
    Args:
        simulation_id: ID of the simulation
        message: Progress message string
        iteration: Current iteration number (for EnKF)
        status: Current status ('processing', 'calibrating', 'forecasting', etc.)
    """
    channel_layer = get_channel_layer()
    if not channel_layer:
        # Channels layer not configured; silently skip broadcasting
        return

    room_group_name = f'simulation_{simulation_id}_progress'
    try:
        await channel_layer.group_send(
            room_group_name,
            {
                'type': 'simulation_progress',
                'message': message,
                'iteration': iteration,
                'status': status,
                'timestamp': datetime.now().isoformat(),
            }
        )
    except Exception:
        # Fail silently if broadcasting is not available (e.g., channels not installed)
        return


async def broadcast_simulation_complete(simulation_id, match_quality=None, best_iteration=None, duration=None):
    """
    Broadcast a completion message to all connected WebSocket clients.
    
    Args:
        simulation_id: ID of the simulation
        match_quality: Final match quality percentage
        best_iteration: Best iteration number
        duration: Total duration in seconds
    """
    channel_layer = get_channel_layer()
    if not channel_layer:
        return

    room_group_name = f'simulation_{simulation_id}_progress'
    try:
        await channel_layer.group_send(
            room_group_name,
            {
                'type': 'simulation_complete',
                'message': 'Simulation completed successfully',
                'match_quality': match_quality,
                'best_iteration': best_iteration,
                'duration': duration,
            }
        )
    except Exception:
        return


async def broadcast_simulation_error(simulation_id, error_message, stack_trace=None):
    """
    Broadcast an error message to all connected WebSocket clients.
    
    Args:
        simulation_id: ID of the simulation
        error_message: Error message string
        stack_trace: Optional stack trace for debugging
    """
    channel_layer = get_channel_layer()
    if not channel_layer:
        return

    room_group_name = f'simulation_{simulation_id}_progress'
    try:
        await channel_layer.group_send(
            room_group_name,
            {
                'type': 'simulation_error',
                'error': error_message,
                'stack_trace': stack_trace,
            }
        )
    except Exception:
        return


def run_async_broadcast(coro):
    """
    Helper function to run async broadcasts from sync code.
    Usage:
        from asgiref.sync import async_to_sync
        async_to_sync(broadcast_simulation_progress)(sim_id, 'message', iteration)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
