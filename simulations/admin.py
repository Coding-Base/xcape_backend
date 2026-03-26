from django.contrib import admin
from simulations.models import Dataset, SimulationRun, Forecast, SimulationStatistics


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    """Dataset admin"""
    
    list_display = ['name', 'user', 'filename', 'file_size', 'uploaded_at']
    list_filter = ['uploaded_at', 'user']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['uploaded_at', 'updated_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'name', 'description')
        }),
        ('File Information', {
            'fields': ('file', 'filename', 'file_size')
        }),
        ('Data', {
            'fields': ('production_data',)
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(SimulationRun)
class SimulationRunAdmin(admin.ModelAdmin):
    """Simulation run admin"""
    
    list_display = ['name', 'user', 'matching_type', 'status', 'progress', 'created_at']
    list_filter = ['status', 'matching_type', 'created_at', 'user']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['created_at', 'started_at', 'completed_at', 'progress', 'match_quality']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'name', 'description', 'dataset')
        }),
        ('Configuration', {
            'fields': ('matching_type', 'initial_pressure', 'porosity', 'permeability', 'water_saturation')
        }),
        ('Status & Progress', {
            'fields': ('status', 'progress', 'current_step', 'total_steps')
        }),
        ('Results', {
            'fields': ('match_quality', 'error_message', 'results_data'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'completed_at', 'duration_seconds'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    """Forecast admin"""
    
    list_display = ['name', 'user', 'simulation', 'forecast_type', 'generated_at']
    list_filter = ['forecast_type', 'generated_at', 'user']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['generated_at', 'updated_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'simulation', 'name', 'description')
        }),
        ('Configuration', {
            'fields': ('forecast_type', 'forecast_date', 'forecast_period_days')
        }),
        ('Results', {
            'fields': ('predicted_parameters', 'predictions', 'uncertainty_bounds'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('generated_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(SimulationStatistics)
class SimulationStatisticsAdmin(admin.ModelAdmin):
    """Simulation statistics admin"""
    
    list_display = ['user', 'total_simulations', 'completed_simulations', 'best_match_quality']
    list_filter = ['updated_at']
    search_fields = ['user__username']
    readonly_fields = ['updated_at']
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Simulation Counts', {
            'fields': ('total_simulations', 'completed_simulations', 'failed_simulations',
                      'baseline_simulations', 'enkf_simulations')
        }),
        ('Data Statistics', {
            'fields': ('total_datasets_uploaded', 'total_forecasts_generated')
        }),
        ('Performance', {
            'fields': ('avg_match_quality', 'avg_simulation_duration', 'best_match_quality',
                      'last_simulation_date')
        }),
    )
    
    def has_add_permission(self, request):
        return False
