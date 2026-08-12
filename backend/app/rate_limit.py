"""
Shared rate-limiter instance for the API.

Defined in its own module to avoid a circular import between ``app.main``
(which registers the limiter middleware) and ``app.api.routes`` (which
applies per-route limits via decorators). Both import the limiter from here.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Key on client IP — appropriate for an unauthenticated API. Behind a reverse
# proxy, ensure X-Forwarded-For is trusted/normalized before this layer.
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
