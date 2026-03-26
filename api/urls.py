from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.views import (
    AuthenticationViewSet, UserViewSet, DashboardViewSet,
    DatasetViewSet, SimulationRunViewSet, ForecastViewSet
)

router = DefaultRouter()
router.register(r'auth', AuthenticationViewSet, basename='auth')
router.register(r'users', UserViewSet, basename='user')
router.register(r'dashboard', DashboardViewSet, basename='dashboard')
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'simulations', SimulationRunViewSet, basename='simulation')
router.register(r'forecasts', ForecastViewSet, basename='forecast')

urlpatterns = [
    path('', include(router.urls)),
]
