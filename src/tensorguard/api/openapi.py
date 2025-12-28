"""
TensorGuard OpenAPI Schema Export

Provides JSON Schema definitions for all API types.
This enables external integrations and documentation generation.
"""

import json
from dataclasses import fields, is_dataclass
from typing import get_type_hints, List, Optional, Dict, Any, Union
from datetime import datetime
import numpy as np

from .schemas import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus


def _python_type_to_json_schema(py_type) -> dict:
    """Convert Python types to JSON Schema types."""
    # Handle Optional types
    origin = getattr(py_type, '__origin__', None)
    
    if origin is Union:
        args = py_type.__args__
        # Optional[X] is Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return {**_python_type_to_json_schema(non_none[0]), "nullable": True}
    
    if origin is list or origin is List:
        item_type = py_type.__args__[0] if py_type.__args__ else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}
    
    if origin is dict or origin is Dict:
        return {"type": "object", "additionalProperties": True}
    
    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "binary"},
        datetime: {"type": "string", "format": "date-time"},
        np.ndarray: {"type": "array", "items": {"type": "number"}},
        Any: {"type": "object"},
    }
    
    return type_map.get(py_type, {"type": "object"})


def dataclass_to_json_schema(dc_class) -> dict:
    """Convert a dataclass to JSON Schema."""
    if not is_dataclass(dc_class):
        raise ValueError(f"{dc_class} is not a dataclass")
    
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": dc_class.__name__,
        "description": dc_class.__doc__ or "",
        "type": "object",
        "properties": {},
        "required": []
    }
    
    type_hints = get_type_hints(dc_class)
    
    for f in fields(dc_class):
        py_type = type_hints.get(f.name, Any)
        prop_schema = _python_type_to_json_schema(py_type)
        
        # Add field description if available
        if f.metadata.get("description"):
            prop_schema["description"] = f.metadata["description"]
        
        # Add default value if present
        if f.default is not f.default_factory:
            if f.default is not None and not callable(f.default):
                prop_schema["default"] = f.default
        
        schema["properties"][f.name] = prop_schema
        
        # Mark as required if no default
        if f.default is f.default_factory and f.default_factory is f.default_factory:
            schema["required"].append(f.name)
    
    return schema


def get_openapi_spec() -> dict:
    """Generate full OpenAPI 3.0 specification for TensorGuard API."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "TensorGuard API",
            "description": "Privacy-preserving federated learning for VLA robotics",
            "version": "2.0.0",
            "contact": {
                "name": "TensorGuard Team",
                "url": "https://github.com/Danielfoojunwei/TensorGuard"
            }
        },
        "components": {
            "schemas": {
                "ShieldConfig": dataclass_to_json_schema(ShieldConfig),
                "Demonstration": dataclass_to_json_schema(Demonstration),
                "SubmissionReceipt": dataclass_to_json_schema(SubmissionReceipt),
                "ClientStatus": dataclass_to_json_schema(ClientStatus),
            }
        },
        "paths": {
            "/api/status": {
                "get": {
                    "summary": "Get dashboard status",
                    "responses": {
                        "200": {
                            "description": "System status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ClientStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/start": {
                "post": {
                    "summary": "Start training simulation",
                    "responses": {
                        "200": {"description": "Simulation started"}
                    }
                }
            },
            "/api/stop": {
                "post": {
                    "summary": "Stop training simulation",
                    "responses": {
                        "200": {"description": "Simulation stopped"}
                    }
                }
            },
            "/api/generate_key": {
                "post": {
                    "summary": "Generate new encryption key",
                    "responses": {
                        "200": {"description": "Key generated"}
                    }
                }
            }
        }
    }


def export_schemas(output_path: str = "openapi.json"):
    """Export OpenAPI spec to file."""
    spec = get_openapi_spec()
    with open(output_path, 'w') as f:
        json.dump(spec, f, indent=2, default=str)
    print(f"OpenAPI spec exported to: {output_path}")
    return spec


# CLI entry point
if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    export_schemas(output)
