from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate, get_user_model
from django.db.models import Q
from django.utils import timezone
from datetime import timedelta
import logging
import numpy as np

from users.models import CustomUser, UserActivityLog
from simulations.models import Dataset, SimulationRun, Forecast, SimulationStatistics
from api.serializers import (
    UserRegistrationSerializer, UserSerializer, UserDetailSerializer,
    DatasetSerializer, SimulationRunCreateSerializer, SimulationRunDetailSerializer,
    SimulationRunListSerializer, ForecastSerializer, DashboardSummarySerializer,
    UserActivityLogSerializer
)

from simulator.forecast_generator import ForecastGenerator
from simulator.enkf_filter import EnKFFilter
from simulator.engine import SimulationEngine

User = get_user_model()

logger = logging.getLogger(__name__)


class IsAuthenticated(permissions.BasePermission):
    """Custom permission to check if user is authenticated"""
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated)


# ============ Authentication Views ============

class AuthenticationViewSet(viewsets.ViewSet):
    """Handle user registration, login, and logout"""
    
    permission_classes = [permissions.AllowAny]
    
    @action(detail=False, methods=['post'])
    def register(self, request):
        """Register a new user account"""
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            # Create token for new user
            token, created = Token.objects.get_or_create(user=user)
            # Create statistics record
            SimulationStatistics.objects.create(user=user)
            
            return Response({
                'user': UserSerializer(user).data,
                'token': token.key,
                'message': 'Account created successfully'
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def login(self, request):
        """Login with username/email and password"""
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({
                'error': 'Please provide both username/email and password'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Try to authenticate with username, then email
        user = User.objects.filter(
            Q(username=username) | Q(email=username)
        ).first()
        
        if user and user.check_password(password):
            # Get or create token
            token, created = Token.objects.get_or_create(user=user)
            
            # Update last login
            user.last_login = timezone.now()
            user.save()
            
            # Log activity
            UserActivityLog.objects.create(
                user=user,
                activity_type='login',
                ip_address=self.get_client_ip(request)
            )
            
            return Response({
                'user': UserDetailSerializer(user, context={'request': request}).data,
                'token': token.key,
                'message': 'Login successful'
            }, status=status.HTTP_200_OK)
        
        return Response({
            'error': 'Invalid credentials'
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def logout(self, request):
        """Logout current user"""
        UserActivityLog.objects.create(
            user=request.user,
            activity_type='logout',
            ip_address=self.get_client_ip(request)
        )
        request.user.auth_token.delete()
        return Response({
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def refresh_token(self, request):
        """Generate a new authentication token"""
        user = request.user
        token, created = Token.objects.get_or_create(user=user)
        if not created:
            token.delete()
            token = Token.objects.create(user=user)
        
        return Response({
            'token': token.key,
            'message': 'Token refreshed'
        }, status=status.HTTP_200_OK)
    
    @staticmethod
    def get_client_ip(request):
        """Get client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


# ============ User Profile Views ============

class UserViewSet(viewsets.ModelViewSet):
    """User profile management"""
    
    permission_classes = [IsAuthenticated]
    serializer_class = UserDetailSerializer
    
    def get_queryset(self):
        """Users can only view their own profile"""
        return User.objects.filter(id=self.request.user.id)
    
    @action(detail=False, methods=['get'])
    def me(self, request):
        """Get current user profile"""
        user = request.user
        serializer = UserDetailSerializer(user, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['put', 'patch'])
    def update_profile(self, request):
        """Update current user profile"""
        user = request.user
        serializer = UserSerializer(user, data=request.data, partial=True,
                                   context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def change_password(self, request):
        """Change user password"""
        user = request.user
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')
        
        if not user.check_password(old_password):
            return Response({
                'error': 'Current password is incorrect'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if len(new_password) < 8:
            return Response({
                'error': 'New password must be at least 8 characters'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user.set_password(new_password)
        user.save()
        return Response({
            'message': 'Password changed successfully'
        }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['get'])
    def activity_log(self, request):
        """Get user activity log"""
        activities = request.user.activity_logs.all()[:50]
        serializer = UserActivityLogSerializer(activities, many=True)
        return Response(serializer.data)


# ============ Dashboard Views ============

class DashboardViewSet(viewsets.ViewSet):
    """User dashboard with statistics and recent activity"""
    
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get complete dashboard summary"""
        user = request.user
        
        # Get statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=user)
        
        # Get recent data
        recent_simulations = user.simulation_runs.all()[:5]
        recent_forecasts = user.forecasts.all()[:5]
        recent_datasets = user.datasets.all()[:5]
        
        data = {
            'user': UserSerializer(user, context={'request': request}).data,
            'statistics': {
                'total_simulations': stats.total_simulations,
                'completed_simulations': stats.completed_simulations,
                'failed_simulations': stats.failed_simulations,
                'baseline_simulations': stats.baseline_simulations,
                'enkf_simulations': stats.enkf_simulations,
                'total_datasets_uploaded': stats.total_datasets_uploaded,
                'total_forecasts_generated': stats.total_forecasts_generated,
                'avg_match_quality': stats.avg_match_quality,
                'avg_simulation_duration': stats.avg_simulation_duration,
                'best_match_quality': stats.best_match_quality,
                'last_simulation_date': stats.last_simulation_date,
            },
            'recent_simulations': SimulationRunListSerializer(recent_simulations, many=True).data,
            'recent_forecasts': ForecastSerializer(recent_forecasts, many=True).data,
            'recent_datasets': DatasetSerializer(recent_datasets, many=True,
                                               context={'request': request}).data,
        }
        
        return Response(data, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get user statistics"""
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        from api.serializers import SimulationStatisticsSerializer
        serializer = SimulationStatisticsSerializer(stats)
        return Response(serializer.data)


# ============ Simulation Data Views ============

class DatasetViewSet(viewsets.ModelViewSet):
    """Manage user datasets"""
    
    permission_classes = [IsAuthenticated]
    serializer_class = DatasetSerializer
    
    def get_queryset(self):
        """Users can only see their own datasets"""
        return Dataset.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Attach current user to dataset"""
        serializer.save(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get 10 most recent datasets"""
        datasets = request.user.datasets.all()[:10]
        serializer = self.get_serializer(datasets, many=True)
        return Response(serializer.data)


class SimulationRunViewSet(viewsets.ModelViewSet):
    """Manage simulation runs"""
    
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Users can only see their own simulations"""
        return SimulationRun.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        """Use different serializer based on action"""
        if self.action == 'create':
            return SimulationRunCreateSerializer
        elif self.action == 'retrieve':
            return SimulationRunDetailSerializer
        return SimulationRunListSerializer
    
    def perform_create(self, serializer):
        """Attach current user to simulation"""
        simulation = serializer.save(user=self.request.user)
        # Log activity
        UserActivityLog.objects.create(
            user=self.request.user,
            activity_type='simulation_start',
            description=f"Started {simulation.matching_type} simulation: {simulation.name}"
        )
    
    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Start/resume a simulation"""
        simulation = self.get_object()
        if simulation.status in ['completed', 'failed']:
            return Response({
                'error': 'Cannot start a completed or failed simulation'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        simulation.status = 'running'
        simulation.started_at = timezone.now()
        simulation.save()
        
        return Response({
            'message': 'Simulation started',
            'simulation': SimulationRunDetailSerializer(simulation).data
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        """Update simulation progress"""
        simulation = self.get_object()
        progress = request.data.get('progress', 0)
        current_step = request.data.get('current_step', 0)
        
        simulation.progress = min(100, progress)
        simulation.current_step = current_step
        simulation.save()
        
        return Response({
            'progress': simulation.progress,
            'current_step': simulation.current_step
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Mark simulation as completed"""
        simulation = self.get_object()
        match_quality = request.data.get('match_quality')
        results_data = request.data.get('results_data', {})
        
        simulation.status = 'completed'
        simulation.progress = 100
        simulation.completed_at = timezone.now()
        simulation.match_quality = match_quality
        simulation.results_data = results_data
        
        if simulation.started_at:
            delta = simulation.completed_at - simulation.started_at
            simulation.duration_seconds = int(delta.total_seconds())
        
        simulation.save()
        
        # Update statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        stats.total_simulations += 1
        stats.completed_simulations += 1
        if simulation.matching_type == 'baseline':
            stats.baseline_simulations += 1
        else:
            stats.enkf_simulations += 1
        
        if match_quality:
            stats.best_match_quality = max(stats.best_match_quality, match_quality)
            # Update average
            if stats.total_simulations > 0:
                total_quality = stats.avg_match_quality * (stats.total_simulations - 1) + match_quality
                stats.avg_match_quality = total_quality / stats.total_simulations
        
        stats.last_simulation_date = timezone.now()
        stats.save()
        
        # Log activity
        UserActivityLog.objects.create(
            user=request.user,
            activity_type='simulation_complete',
            description=f"Completed simulation: {simulation.name} (Match Quality: {match_quality}%)"
        )
        
        return Response({
            'message': 'Simulation completed',
            'simulation': SimulationRunDetailSerializer(simulation).data
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def fail(self, request, pk=None):
        """Mark simulation as failed"""
        simulation = self.get_object()
        error_message = request.data.get('error_message', 'Unknown error')
        
        simulation.status = 'failed'
        simulation.error_message = error_message
        simulation.completed_at = timezone.now()
        simulation.save()
        
        # Update statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        stats.total_simulations += 1
        stats.failed_simulations += 1
        stats.save()
        
        return Response({
            'message': 'Simulation marked as failed',
            'simulation': SimulationRunDetailSerializer(simulation).data
        }, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def generate_forecast(self, request, pk=None):
        """Generate a prior or posterior forecast for a simulation"""
        simulation = self.get_object()
        forecast_type = request.data.get('forecast_type', 'prior')
        forecast_period_days = int(request.data.get('forecast_period_days', 365))
        # Optional: accept ensemble parameters from client
        ensemble_params = request.data.get('ensemble_params')

        fg = ForecastGenerator()

        # Simple forward model using synthetic production data (fallback)
        def forward_model_fn(params):
            return {'production_data': fg._get_synthetic_production_data(forecast_period_days)}

        # If no ensemble provided, generate a small ensemble from simulation parameters
        if not ensemble_params:
            base_params = {
                'initial_pressure': simulation.initial_pressure,
                'porosity': simulation.porosity,
                'permeability': simulation.permeability,
                'water_saturation': simulation.water_saturation,
            }
            # create simple ensemble with small perturbations
            ensemble_params = []
            for i in range(10):
                perturbed = {k: float(v) * (1 + 0.02 * (i - 5)) for k, v in base_params.items()}
                ensemble_params.append(perturbed)

        result = fg.generate_forecast(
            str(simulation.id),
            ensemble_params,
            forward_model_fn,
            forecast_type=forecast_type,
            forecast_period_days=forecast_period_days
        )

        if result.get('status') != 'completed':
            return Response({'error': result.get('error', 'Failed to generate forecast')},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        forecast_payload = result['forecast']

        # Persist Forecast model
        forecast_obj = Forecast.objects.create(
            simulation=simulation,
            user=request.user,
            name=f"{simulation.name} - {forecast_type} forecast",
            description=request.data.get('description', ''),
            forecast_type=forecast_type,
            forecast_date=request.data.get('forecast_date') or None,
            forecast_period_days=forecast_period_days,
            predicted_parameters=forecast_payload.get('predictions', {}),
            predictions=forecast_payload.get('predictions', {}),
            uncertainty_bounds=forecast_payload.get('uncertainty', {})
        )

        # Update user statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
        stats.total_forecasts_generated += 1
        stats.save()

        serializer = ForecastSerializer(forecast_obj, context={'request': request})
        return Response({'message': 'Forecast generated', 'forecast': serializer.data},
                        status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def compare_forecasts(self, request, pk=None):
        """Generate prior/posterior comparison using provided ensembles or EnKF results"""
        simulation = self.get_object()
        prior_ensemble = request.data.get('prior_ensemble')
        posterior_ensemble = request.data.get('posterior_ensemble')
        forecast_period_days = int(request.data.get('forecast_period_days', 365))

        fg = ForecastGenerator()

        def forward_model_fn(params):
            return {'production_data': fg._get_synthetic_production_data(forecast_period_days)}

        if not prior_ensemble or not posterior_ensemble:
            # Attempt to use simulation.results_data to derive ensembles (if EnKF ran)
            results = simulation.results_data or {}
            prior_ensemble = prior_ensemble or results.get('prior_ensemble')
            posterior_ensemble = posterior_ensemble or results.get('posterior_ensemble')

        if not prior_ensemble or not posterior_ensemble:
            return Response({'error': 'Both prior_ensemble and posterior_ensemble are required'},
                            status=status.HTTP_400_BAD_REQUEST)

        comparison = fg.generate_prior_posterior_comparison(
            str(simulation.id), prior_ensemble, posterior_ensemble, forward_model_fn, forecast_period_days
        )

        if comparison.get('status') != 'completed':
            return Response({'error': comparison.get('error', 'Failed to generate comparison')},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({'message': 'Comparison generated', 'comparison': comparison}, status=status.HTTP_200_OK)

    @action(detail=True, methods=['post'])
    def run_enkf_with_forecasts(self, request, pk=None):
        """Run complete EnKF pipeline with prior and posterior forecasts"""
        simulation = self.get_object()
        
        # Get parameters from simulation or request
        ensemble_size = int(request.data.get('ensemble_size', 100))
        num_iterations = int(request.data.get('num_iterations', 10))
        forecast_period_days = int(request.data.get('forecast_period_days', 365))
        
        try:
            # Initialize EnKF
            enkf = EnKFFilter(ensemble_size=ensemble_size)
            
            # Create initial ensemble from simulation parameters
            mean_params = {
                'initial_pressure': simulation.initial_pressure,
                'porosity': simulation.porosity,
                'permeability': simulation.permeability,
                'water_saturation': simulation.water_saturation,
            }
            
            prior_ensemble = enkf.initialize_ensemble(mean_params)
            
            # Simple forward model for testing (would use real OPM in production)
            def forward_model_fn(params):
                fg = ForecastGenerator()
                return {'production_data': fg._get_synthetic_production_data(100)}
            
            # Run EnKF
            def progress_cb(progress, msg):
                simulation.progress = int(progress)
                simulation.current_step += 1
                simulation.save()
            
            enkf_result = enkf.run_enkf(
                observed_data={'production_data': {}},
                forward_model_fn=forward_model_fn,
                initial_ensemble=prior_ensemble,
                num_iterations=num_iterations,
                progress_callback=progress_cb
            )
            
            if enkf_result.get('status') != 'completed':
                return Response({'error': 'EnKF run failed'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Extract posterior ensemble
            posterior_ensemble = np.array(enkf_result['final_ensemble']).tolist()
            
            # Generate PRIOR forecast (before EnKF)
            fg = ForecastGenerator()
            prior_result = fg.generate_forecast(
                str(simulation.id) + '_prior',
                prior_ensemble.tolist(),
                forward_model_fn,
                'prior',
                forecast_period_days
            )
            
            # Generate POSTERIOR forecast (after EnKF)
            posterior_result = fg.generate_forecast(
                str(simulation.id) + '_posterior',
                posterior_ensemble,
                forward_model_fn,
                'posterior',
                forecast_period_days
            )
            
            # Persist both forecasts
            forecasts = []
            
            if prior_result.get('status') == 'completed':
                prior_forecast = Forecast.objects.create(
                    simulation=simulation,
                    user=request.user,
                    name=f"{simulation.name} - Prior Forecast",
                    description="Prior forecast before EnKF calibration",
                    forecast_type='prior',
                    forecast_date=request.data.get('forecast_date') or None,
                    forecast_period_days=forecast_period_days,
                    predicted_parameters=prior_result['forecast'].get('predictions', {}),
                    predictions=prior_result['forecast'].get('predictions', {}),
                    uncertainty_bounds=prior_result['forecast'].get('uncertainty', {})
                )
                forecasts.append(ForecastSerializer(prior_forecast).data)
            
            if posterior_result.get('status') == 'completed':
                posterior_forecast = Forecast.objects.create(
                    simulation=simulation,
                    user=request.user,
                    name=f"{simulation.name} - Posterior Forecast",
                    description="Posterior forecast after EnKF calibration",
                    forecast_type='posterior',
                    forecast_date=request.data.get('forecast_date') or None,
                    forecast_period_days=forecast_period_days,
                    predicted_parameters=posterior_result['forecast'].get('predictions', {}),
                    predictions=posterior_result['forecast'].get('predictions', {}),
                    uncertainty_bounds=posterior_result['forecast'].get('uncertainty', {})
                )
                forecasts.append(ForecastSerializer(posterior_forecast).data)
            
            # Update simulation with EnKF results
            simulation.status = 'completed'
            simulation.match_quality = float(enkf_result.get('best_quality', 0))
            simulation.progress = 100
            simulation.results_data = {
                'enkf_result': {k: v for k, v in enkf_result.items() if k != 'final_ensemble'},
                'prior_ensemble': prior_ensemble.tolist() if isinstance(prior_ensemble, np.ndarray) else prior_ensemble,
                'posterior_ensemble': posterior_ensemble,
            }
            simulation.completed_at = timezone.now()
            if simulation.started_at:
                delta = simulation.completed_at - simulation.started_at
                simulation.duration_seconds = int(delta.total_seconds())
            simulation.save()
            
            # Update statistics
            stats, _ = SimulationStatistics.objects.get_or_create(user=request.user)
            stats.total_simulations += 1
            stats.completed_simulations += 1
            stats.enkf_simulations += 1
            stats.total_forecasts_generated += 2  # prior + posterior
            stats.best_match_quality = max(stats.best_match_quality, simulation.match_quality)
            stats.save()
            
            return Response({
                'message': 'EnKF with forecasts completed',
                'simulation': SimulationRunDetailSerializer(simulation).data,
                'forecasts': forecasts,
                'enkf_quality': float(enkf_result.get('best_quality', 0))
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"EnKF pipeline failed: {e}")
            simulation.status = 'failed'
            simulation.error_message = str(e)
            simulation.save()
            return Response({'error': str(e)}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get 10 most recent simulations"""
        simulations = request.user.simulation_runs.all()[:10]
        serializer = SimulationRunListSerializer(simulations, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def completed(self, request):
        """Get all completed simulations"""
        simulations = request.user.simulation_runs.filter(status='completed')
        serializer = SimulationRunListSerializer(simulations, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_type(self, request):
        """Get simulations by type"""
        matching_type = request.query_params.get('type', 'baseline')
        simulations = request.user.simulation_runs.filter(matching_type=matching_type)
        serializer = SimulationRunListSerializer(simulations, many=True)
        return Response(serializer.data)


class ForecastViewSet(viewsets.ModelViewSet):
    """Manage forecasts from simulations"""
    
    permission_classes = [IsAuthenticated]
    serializer_class = ForecastSerializer
    
    def get_queryset(self):
        """Users can only see their own forecasts"""
        return Forecast.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        """Attach current user to forecast"""
        forecast = serializer.save(user=self.request.user)
        
        # Update statistics
        stats, _ = SimulationStatistics.objects.get_or_create(user=self.request.user)
        stats.total_forecasts_generated += 1
        stats.save()
        
        # Log activity
        UserActivityLog.objects.create(
            user=self.request.user,
            activity_type='forecast_generated',
            description=f"Generated forecast: {forecast.name}"
        )
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get 10 most recent forecasts"""
        forecasts = request.user.forecasts.all()[:10]
        serializer = self.get_serializer(forecasts, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_simulation(self, request):
        """Get forecasts for a specific simulation"""
        simulation_id = request.query_params.get('simulation_id')
        if not simulation_id:
            return Response({
                'error': 'simulation_id parameter required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        forecasts = request.user.forecasts.filter(simulation_id=simulation_id)
        serializer = self.get_serializer(forecasts, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def parameter_histogram(self, request):
        """Return histogram data for a parameter across an ensemble or forecasts"""
        import numpy as np

        simulation_id = request.query_params.get('simulation_id')
        parameter = request.query_params.get('parameter')
        bins = int(request.query_params.get('bins', 20))

        if not simulation_id or not parameter:
            return Response({'error': 'simulation_id and parameter are required'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Try to collect parameter values from forecasts' predicted_parameters
        forecasts = self.request.user.forecasts.filter(simulation_id=simulation_id)
        values = []
        for f in forecasts:
            params = f.predicted_parameters or {}
            if isinstance(params, dict) and parameter in params:
                try:
                    values.append(float(params[parameter]))
                except Exception:
                    continue

        if not values:
            return Response({'error': 'No parameter values found for histogram'},
                            status=status.HTTP_404_NOT_FOUND)

        arr = np.array(values)
        counts, edges = np.histogram(arr, bins=bins)
        return Response({
            'parameter': parameter,
            'counts': counts.tolist(),
            'bin_edges': edges.tolist(),
            'n': int(arr.size)
        })

    @action(detail=False, methods=['get'])
    def comparison_chart(self, request):
        """Return series data (mean/p10/p50/p90) for prior vs posterior for a metric"""
        simulation_id = request.query_params.get('simulation_id')
        metric = request.query_params.get('metric', 'oil')

        if not simulation_id:
            return Response({'error': 'simulation_id required'}, status=status.HTTP_400_BAD_REQUEST)

        # Find latest prior and posterior forecasts
        prior = self.request.user.forecasts.filter(simulation_id=simulation_id, forecast_type='prior').order_by('-generated_at').first()
        posterior = self.request.user.forecasts.filter(simulation_id=simulation_id, forecast_type='posterior').order_by('-generated_at').first()

        if not prior or not posterior:
            return Response({'error': 'Both prior and posterior forecasts required'}, status=status.HTTP_404_NOT_FOUND)

        def extract_series(forecast_obj, metric_key):
            preds = forecast_obj.predictions or {}
            metric_data = preds.get(metric_key, {})
            return {
                'mean': metric_data.get('mean', []),
                'p10': metric_data.get('p10', []),
                'p50': metric_data.get('p50', []),
                'p90': metric_data.get('p90', []),
            }

        data = {
            'simulation_id': simulation_id,
            'metric': metric,
            'prior': extract_series(prior, metric),
            'posterior': extract_series(posterior, metric),
            'time_axis': prior.predictions.get('time_axis') if prior.predictions else None
        }

        return Response(data)
