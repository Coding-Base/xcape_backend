from rest_framework import serializers
from django.contrib.auth import get_user_model
from users.models import UserActivityLog, CustomUser
from simulations.models import Dataset, SimulationRun, Forecast, SimulationStatistics

User = get_user_model()


# ============ User Serializers ============

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password2 = serializers.CharField(write_only=True, min_length=8)
    
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'first_name', 'last_name', 'password', 'password2',
                  'institution', 'department']
    
    def validate(self, data):
        if data['password'] != data.pop('password2'):
            raise serializers.ValidationError({"password": "Passwords do not match."})
        return data
    
    def create(self, validated_data):
        user = CustomUser.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            password=validated_data['password'],
            institution=validated_data.get('institution', ''),
            department=validated_data.get('department', ''),
        )
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone',
                  'profile_image', 'bio', 'institution', 'department', 'research_area',
                  'years_experience', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class UserDetailSerializer(serializers.ModelSerializer):
    statistics = serializers.SerializerMethodField()
    recent_simulations = serializers.SerializerMethodField()
    
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone',
                  'profile_image', 'bio', 'institution', 'department', 'research_area',
                  'years_experience', 'date_joined', 'last_login', 'statistics',
                  'recent_simulations']
        read_only_fields = ['id', 'date_joined', 'last_login']
    
    def get_statistics(self, obj):
        try:
            stats = obj.statistics
            return SimulationStatisticsSerializer(stats).data
        except SimulationStatistics.DoesNotExist:
            return None
    
    def get_recent_simulations(self, obj):
        recent = obj.simulation_runs.all()[:5]
        return SimulationRunListSerializer(recent, many=True).data


class UserActivityLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserActivityLog
        fields = ['id', 'activity_type', 'description', 'timestamp']
        read_only_fields = fields


# ============ Simulation Data Serializers ============

class DatasetSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    file = serializers.FileField(write_only=True, required=True)
    
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'file', 'filename', 'file_size', 'uploaded_at',
                  'updated_at', 'file_url', 'production_data']
        read_only_fields = ['id', 'filename', 'file_size', 'uploaded_at', 'updated_at']
    
    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None
    
    def create(self, validated_data):
        file = validated_data.pop('file')
        dataset = Dataset.objects.create(
            file=file,
            filename=file.name,
            file_size=file.size,
            **validated_data
        )
        return dataset


class SimulationRunListSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = SimulationRun
        fields = ['id', 'name', 'matching_type', 'status', 'progress', 'match_quality',
                  'created_at', 'completed_at', 'dataset_name', 'duration_seconds']
        read_only_fields = ['id', 'created_at', 'completed_at']


class SimulationRunDetailSerializer(serializers.ModelSerializer):
    dataset = DatasetSerializer(read_only=True)
    forecasts = serializers.SerializerMethodField()
    
    class Meta:
        model = SimulationRun
        fields = ['id', 'name', 'description', 'matching_type', 'status', 'dataset',
                  'initial_pressure', 'porosity', 'permeability', 'water_saturation',
                  'progress', 'match_quality', 'error_message', 'results_data',
                  'created_at', 'started_at', 'completed_at', 'duration_seconds',
                  'forecasts']
        read_only_fields = ['id', 'progress', 'match_quality', 'error_message',
                           'results_data', 'created_at', 'started_at', 'completed_at',
                           'duration_seconds']
    
    def get_forecasts(self, obj):
        forecasts = obj.forecasts.all()
        return ForecastSerializer(forecasts, many=True).data


class SimulationRunCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationRun
        fields = ['id', 'name', 'description', 'matching_type', 'dataset',
                  'initial_pressure', 'porosity', 'permeability', 'water_saturation', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)


class ForecastSerializer(serializers.ModelSerializer):
    class Meta:
        model = Forecast
        fields = ['id', 'name', 'description', 'forecast_type', 'forecast_date',
                  'forecast_period_days', 'predicted_parameters', 'predictions',
                  'uncertainty_bounds', 'generated_at']
        read_only_fields = ['id', 'generated_at']


class SimulationStatisticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationStatistics
        fields = ['total_simulations', 'completed_simulations', 'failed_simulations',
                  'baseline_simulations', 'enkf_simulations', 'total_datasets_uploaded',
                  'total_forecasts_generated', 'avg_match_quality', 'avg_simulation_duration',
                  'best_match_quality', 'last_simulation_date']
        read_only_fields = fields


# ============ Dashboard Serializers ============

class DashboardSummarySerializer(serializers.Serializer):
    """Combines user info with statistics for dashboard view"""
    user = UserSerializer()
    statistics = SimulationStatisticsSerializer()
    recent_simulations = SimulationRunListSerializer(many=True)
    recent_forecasts = ForecastSerializer(many=True)
    recent_datasets = DatasetSerializer(many=True)
    
    def to_representation(self, instance):
        """Custom representation to handle dict/object mixing"""
        return instance
