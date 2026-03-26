from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from users.models import CustomUser, UserActivityLog


@admin.register(CustomUser)
class CustomUserAdmin(BaseUserAdmin):
    """Custom user admin with extended fields"""
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Profile Information', {
            'fields': ('phone', 'profile_image', 'bio')
        }),
        ('Institution & Research', {
            'fields': ('institution', 'department', 'research_area', 'years_experience')
        }),
        ('Preferences', {
            'fields': ('notifications_enabled', 'allow_data_tracking')
        }),
    )
    
    list_display = ['username', 'email', 'first_name', 'last_name', 'institution', 'date_joined']
    list_filter = ['date_joined', 'is_active', 'institution']
    search_fields = ['username', 'email', 'first_name', 'last_name']


@admin.register(UserActivityLog)
class UserActivityLogAdmin(admin.ModelAdmin):
    """Activity log admin"""
    
    list_display = ['user', 'activity_type', 'timestamp', 'ip_address']
    list_filter = ['activity_type', 'timestamp']
    search_fields = ['user__username', 'description']
    readonly_fields = ['user', 'activity_type', 'description', 'ip_address', 'timestamp']
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False
