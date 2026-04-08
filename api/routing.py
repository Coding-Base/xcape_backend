"""
WebSocket URL routing for XCAPE API.
Maps WebSocket connections to consumers.
"""

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # WebSocket route for simulation progress updates
    # Pattern: /ws/simulation/{simulation_id}/progress/
    re_path(
        r'ws/simulation/(?P<simulation_id>\d+)/progress/$',
        consumers.SimulationProgressConsumer.as_asgi()
    ),
]
