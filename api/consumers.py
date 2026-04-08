import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from simulations.models import SimulationRun


class SimulationProgressConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time simulation progress updates.
    
    Clients connect to: ws://localhost:8000/ws/simulation/{simulation_id}/progress/
    """

    async def connect(self):
        """Called when a new WebSocket connection is established."""
        self.simulation_id = self.scope['url_route']['kwargs']['simulation_id']
        self.room_group_name = f'simulation_{self.simulation_id}_progress'

        # Join the room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        # Accept the connection
        await self.accept()
        
        # Send initial message confirming connection
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': f'Connected to simulation {self.simulation_id} progress stream',
            'simulation_id': self.simulation_id,
        }))

    async def disconnect(self, close_code):
        """Called when a WebSocket connection is closed."""
        # Leave the room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """
        Called when the client sends data over WebSocket.
        Currently not used - this is a one-way push channel.
        """
        pass

    # Handler for simulation progress messages
    async def simulation_progress(self, event):
        """
        Called when a message with type 'simulation_progress' is sent to the group.
        Forwards the progress update to the client.
        """
        message = event['message']
        iteration = event.get('iteration', 0)
        status = event.get('status', 'processing')
        
        await self.send(text_data=json.dumps({
            'type': 'progress_update',
            'message': message,
            'iteration': iteration,
            'status': status,
            'timestamp': event.get('timestamp'),
        }))

    # Handler for simulation complete messages
    async def simulation_complete(self, event):
        """
        Called when a message with type 'simulation_complete' is sent to the group.
        """
        print(f"[Consumer] Sending simulation_complete event: {event}")
        await self.send(text_data=json.dumps({
            'type': 'simulation_complete',
            'message': event.get('message', 'Simulation completed'),
            'match_quality': event.get('match_quality'),
            'best_iteration': event.get('best_iteration'),
            'duration': event.get('duration'),
        }))

    # Handler for simulation error messages
    async def simulation_error(self, event):
        """
        Called when a message with type 'simulation_error' is sent to the group.
        """
        await self.send(text_data=json.dumps({
            'type': 'simulation_error',
            'error': event.get('error', 'An error occurred'),
            'stack_trace': event.get('stack_trace'),
        }))
