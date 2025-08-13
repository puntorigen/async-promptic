"""
Django model support for async-promptic.

This module provides utilities to use Django models as output schemas
for LLM-decorated functions, similar to Pydantic model support.
"""

import json
from typing import Any, Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from django.db import models as django_models

# Try to import Django, but don't fail if it's not installed
try:
    from django.db import models
    from django.core.exceptions import ValidationError
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    models = None
    ValidationError = None


def is_django_model(cls: Any) -> bool:
    """Check if a class is a Django model."""
    if not DJANGO_AVAILABLE:
        return False
    
    try:
        return isinstance(cls, type) and issubclass(cls, models.Model)
    except (TypeError, AttributeError):
        return False


def django_field_to_json_type(field) -> Dict[str, Any]:
    """Convert a Django model field to JSON schema type definition."""
    if not DJANGO_AVAILABLE:
        return {"type": "string"}
    
    schema = {}
    
    # Map Django field types to JSON schema types
    if isinstance(field, models.CharField):
        schema["type"] = "string"
        if field.max_length:
            schema["maxLength"] = field.max_length
    elif isinstance(field, models.TextField):
        schema["type"] = "string"
    elif isinstance(field, models.IntegerField):
        schema["type"] = "integer"
    elif isinstance(field, models.FloatField):
        schema["type"] = "number"
    elif isinstance(field, models.DecimalField):
        schema["type"] = "number"
    elif isinstance(field, models.BooleanField):
        schema["type"] = "boolean"
    elif isinstance(field, models.DateField):
        schema["type"] = "string"
        schema["format"] = "date"
    elif isinstance(field, models.DateTimeField):
        schema["type"] = "string"
        schema["format"] = "date-time"
    elif isinstance(field, models.TimeField):
        schema["type"] = "string"
        schema["format"] = "time"
    elif isinstance(field, models.EmailField):
        schema["type"] = "string"
        schema["format"] = "email"
    elif isinstance(field, models.URLField):
        schema["type"] = "string"
        schema["format"] = "uri"
    elif isinstance(field, models.UUIDField):
        schema["type"] = "string"
        schema["format"] = "uuid"
    elif isinstance(field, models.JSONField):
        schema["type"] = "object"
    elif isinstance(field, models.ForeignKey):
        # For foreign keys, we'll expect an ID
        schema["type"] = "integer"
        schema["description"] = f"ID of related {field.related_model.__name__}"
    elif isinstance(field, models.ManyToManyField):
        # For many-to-many, expect an array of IDs
        schema["type"] = "array"
        schema["items"] = {"type": "integer"}
        schema["description"] = f"IDs of related {field.related_model.__name__} objects"
    else:
        # Default fallback
        schema["type"] = "string"
    
    # Add common field attributes
    if hasattr(field, 'help_text') and field.help_text:
        schema["description"] = field.help_text
    
    if hasattr(field, 'choices') and field.choices:
        schema["enum"] = [choice[0] for choice in field.choices]
    
    if hasattr(field, 'default') and field.default != models.NOT_PROVIDED:
        if callable(field.default):
            # Skip callable defaults as they can't be serialized
            pass
        else:
            schema["default"] = field.default
    
    # Handle null/blank
    if getattr(field, 'null', False) or getattr(field, 'blank', False):
        # Make the field optional by not including it in required fields
        pass
    
    return schema


def django_model_to_json_schema(model_class: Type["django_models.Model"]) -> Dict[str, Any]:
    """Convert a Django model to a JSON schema."""
    if not DJANGO_AVAILABLE:
        raise ImportError("Django is not installed")
    
    if not is_django_model(model_class):
        raise ValueError(f"{model_class} is not a Django model")
    
    properties = {}
    required = []
    
    # Iterate through all fields in the model
    for field in model_class._meta.fields:
        # Skip auto-created fields like 'id' unless explicitly defined
        if field.auto_created and not field.primary_key:
            continue
        
        # Skip the auto-generated id field for new instances
        if field.primary_key and field.name == 'id' and getattr(field, 'auto_created', False):
            continue
        
        field_schema = django_field_to_json_type(field)
        properties[field.name] = field_schema
        
        # Add to required if not nullable and no default
        if not field.null and not field.blank and field.default == models.NOT_PROVIDED:
            required.append(field.name)
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "title": model_class.__name__,
    }
    
    # Add model docstring as description if available
    if model_class.__doc__:
        schema["description"] = model_class.__doc__.strip()
    
    return schema


def create_django_model_instance(model_class: Type["django_models.Model"], data: Dict[str, Any]) -> "django_models.Model":
    """
    Create a Django model instance from parsed JSON data.
    
    The instance is NOT saved to the database - that's left to the user.
    """
    if not DJANGO_AVAILABLE:
        raise ImportError("Django is not installed")
    
    if not is_django_model(model_class):
        raise ValueError(f"{model_class} is not a Django model")
    
    # Filter out any fields that don't exist on the model
    model_fields = {f.name for f in model_class._meta.fields}
    filtered_data = {k: v for k, v in data.items() if k in model_fields}
    
    # Handle foreign key fields - they might come as IDs
    for field in model_class._meta.fields:
        if isinstance(field, models.ForeignKey):
            field_name = field.name
            # If we have an ID for this field, rename it to field_id for Django
            if field_name in filtered_data and isinstance(filtered_data[field_name], (int, str)):
                filtered_data[f"{field_name}_id"] = filtered_data.pop(field_name)
    
    # Create the instance (not saved to DB)
    instance = model_class(**filtered_data)
    
    # Validate the instance (but don't save)
    try:
        instance.full_clean()
    except ValidationError as e:
        # You might want to handle or log validation errors
        # For now, we'll just pass through the instance even if validation fails
        # The user can handle validation as needed
        pass
    
    return instance


def get_django_model_instructions(model_class: Type["django_models.Model"]) -> str:
    """
    Generate instructions for the LLM about how to format the Django model response.
    """
    if not is_django_model(model_class):
        return ""
    
    schema = django_model_to_json_schema(model_class)
    
    instructions = f"""
You must return a JSON object that matches this Django model schema:

{json.dumps(schema, indent=2)}

The response should be valid JSON that can be used to create an instance of the {model_class.__name__} Django model.
Make sure all required fields are included and match the expected types.
"""
    
    return instructions
