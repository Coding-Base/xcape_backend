from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.views import (
    AuthenticationViewSet, UserViewSet, DashboardViewSet,
    DatasetViewSet, ForecastViewSet, SimulationRunViewSet,
    SensitivityHealthView, SensitivityDatasetPreviewView, SensitivitySimulationView,
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
    # Sensitivity compatibility endpoints for embedded app
    path('sensitivity/health/', SensitivityHealthView.as_view(), name='sensitivity-health'),
    path('sensitivity/dataset-preview/', SensitivityDatasetPreviewView.as_view(), name='sensitivity-dataset-preview'),
    path('sensitivity/simulate/', SensitivitySimulationView.as_view(), name='sensitivity-simulate'),
]
