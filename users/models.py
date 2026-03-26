from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import URLValidator


class CustomUser(AbstractUser):
    """Extended user model with additional fields for researcher profiles"""
    
    # Basic info
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    profile_image = models.ImageField(upload_to='profile_images/', blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    
    # Institution & Research info
    institution = models.CharField(max_length=255, blank=True, null=True, 
                                   help_text="e.g., University of Lagos")
    department = models.CharField(max_length=255, blank=True, null=True,
                                  help_text="e.g., Petroleum Engineering")
    research_area = models.CharField(max_length=255, blank=True, null=True,
                                    help_text="e.g., Reservoir Simulation")
    years_experience = models.IntegerField(blank=True, null=True,
                                          help_text="Years in the field")
    
    # Account management
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)
    
    # Preferences
    notifications_enabled = models.BooleanField(default=True)
    allow_data_tracking = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'custom_user'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self):
        return f"{self.get_full_name()} ({self.email})"


class UserActivityLog(models.Model):
    """Track user activities for analytics and monitoring"""
    
    ACTIVITY_CHOICES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('simulation_start', 'Simulation Started'),
        ('simulation_complete', 'Simulation Completed'),
        ('dataset_upload', 'Dataset Uploaded'),
        ('forecast_generated', 'Forecast Generated'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='activity_logs')
    activity_type = models.CharField(max_length=50, choices=ACTIVITY_CHOICES)
    description = models.TextField(blank=True, null=True)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'user_activity_log'
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.user.username} - {self.activity_type} - {self.timestamp}"
