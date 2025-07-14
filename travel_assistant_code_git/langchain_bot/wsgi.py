"""
WSGI config for langchain_bot project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "langchain_bot.settings")
# settings.py
# ...
WSGI_APPLICATION = '.wsgi.application'
ASGI_APPLICATION = '.asgi.application' # Add this line
# ...
application = get_wsgi_application()
